# `bench_multi_turn.py` —— SGLang 多轮对话基准测试脚本

基于 SGLang 的多轮对话场景压测工具。相比 `bench_serving.py` 的 `generated-shared-prefix` 数据集，本脚本提供**严格可控的 prompt 长度**、**贴近真实对话的轮次语义**、以及**细粒度的精度审计能力**。

- **脚本位置**：`python/sglang/benchmark/bench_multi_turn.py`
- **调用方式**：`python -m sglang.benchmark.bench_multi_turn [options]`

---

## 目录

1. [设计动机](#1-设计动机)
2. [核心概念](#2-核心概念)
3. [整体架构与数据流](#3-整体架构与数据流)
4. [数据生成模式](#4-数据生成模式)
5. [后端与请求协议](#5-后端与请求协议)
6. [多轮对话语义](#6-多轮对话语义)
7. [并发与调度](#7-并发与调度)
8. [长度控制](#8-长度控制)
9. [CLI 参数完整说明](#9-cli-参数完整说明)
10. [输出文件](#10-输出文件)
11. [指标与汇总字段](#11-指标与汇总字段)
12. [精度验证工作流](#12-精度验证工作流)
13. [兼容性与依赖](#13-兼容性与依赖)
14. [已知限制](#14-已知限制)
15. [常见使用示例](#15-常见使用示例)

---

## 1. 设计动机

原生 `bench_serving.py` 的 `generated-shared-prefix` / `random` / `random-ids` 数据集存在以下实际问题：

| 问题 | 原生脚本表现 | 本脚本解决方案 |
|---|---|---|
| 乱码 token decode 后再 encode，长度严重偏差（实测 2-6x 膨胀） | 指定 14400 → 实际 80000+ | `synthetic-ids` / `sharegpt` 模式全程以 `List[int]` 流转，零膨胀 |
| 多轮对话 `prompt_len=1` 欺骗性汇报 | 客户端侧显示 1，服务端侧爆长 | 每轮汇报期望长度和服务端实测长度的差值 |
| `--random-range-ratio` 默认 0.0 导致长度均匀采样 | 平均值 ≈ 目标值的一半 | 默认 `--length-range-ratio=1.0` 固定长度 |
| 没有按用户切分 prompt 共享语义 | 只能用 group 强 shuffle | 同 group 共享 system_prompt；每个 question 独立采样 |
| 无法区分"用户 A 跑完再换用户 B"和"用户 A 和 B 交错" | 全局 queue + semaphore | 信号量在 conversation 粒度，保证单用户内严格串行 |
| 真实对话数据需要自己拼 | 不支持 | 内置 ShareGPT 模式，按角色分三个池子采样 |
| 精度没法审计 | 无工具 | 可选 dump 生成素材 + 每次请求 payload + 服务端 reported 长度 |

## 2. 核心概念

### Group / Conversation / Turn 三级结构

```
Group（用户） = 一个 system_prompt（只采样一次，同 group 共享）
 ├── Conversation 0 —— 同一个 system_prompt，独立的问答序列
 │    ├── Turn 0 → 独立采样 question + pre-gen answer placeholder
 │    ├── Turn 1 → ...
 │    └── Turn N-1 → ...
 ├── Conversation 1 —— 独立的 question 序列
 │    └── ...
 └── Conversation M-1
Group 1
 └── ...
```

- **一个 group = 一个模拟用户**：`--num-groups` 控制
- **一个 group 里有多个 conversation**：`--prompts-per-group` 控制
- **一个 conversation 包含 N 轮对话**：`--num-turns` 控制
- **总请求数** = `num_groups × prompts_per_group × num_turns`（可被 `--num-prompts` 截断）

### 为什么这样分层？

- **用户维度 (group)** 负责**共享前缀**。真实场景里同一用户的不同请求共享一段系统 prompt，这部分在服务端 prefix cache 里可以高度命中
- **对话维度 (conversation)** 负责**多轮累积**。第 t 轮的 prompt 会带上前 t-1 轮的 Q/A 历史
- **轮次维度 (turn)** 是一次实际的 HTTP 请求

与 `bench_serving.py` 的 GSP 对照：

| GSP 参数 | 本脚本参数 |
|---|---|
| `--gsp-num-groups` | `--num-groups` |
| `--gsp-prompts-per-group` | `--prompts-per-group` |
| `--gsp-system-prompt-len` | `--system-prompt-len` |
| `--gsp-question-len` | `--question-len` |
| `--gsp-output-len` | `--output-len` |
| `--gsp-num-turns` | `--num-turns` |

## 3. 整体架构与数据流

```
┌──────────┐   ┌──────────┐   ┌──────────────────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐
│  CLI     │──▶│  Config  │──▶│   DataGenerator      │──▶│  Scheduler  │──▶│  Sender  │──▶│ Reporter │
│  args    │   │ (valid.) │   │  (预生成所有轮次素材)    │   │  并发+速率   │   │  HTTP    │   │ dump+csv │
└──────────┘   └──────────┘   └──────────────────────┘   └─────────────┘   └──────────┘   └──────────┘
                                      │
                                      ▼
                                List[Conversation]
                                （每个 Conversation 带 system_ids + N 个 TurnMaterials）
```

1. **CLI → Config**：`argparse` → `Config` dataclass，`validate_config` 做互斥校验
2. **Config → DataGenerator**：根据 `--dataset-mode` 实例化三种之一
3. **Generator → Conversations**：一次性预生成所有 `num_groups × prompts_per_group × num_turns` 份素材
   - 每个 group 采样一次 system prompt（同 group 共享）
   - 每个 (group, conv, turn) 独立采样 question + answer placeholder
4. **Scheduler**：异步调度所有 conversation，用 `Semaphore` 限并发，用 Poisson 间隔控到达速率
5. **Sender**：每一轮用当前累积的 prompt（system + Q1 + A1 + ... + Qt）调用 `/generate` 或 `/v1/chat/completions`
6. **Reporter**：汇总指标，写 JSONL / CSV / dump 文件

## 4. 数据生成模式

三种模式用 `--dataset-mode` 切换。核心差异：

| 模式 | 数据来源 | 长度精度 | 支持后端 |
|---|---|---|---|
| `synthetic-ids`（默认） | 随机采样词表 id | **严格精确** | 仅 `sglang` 原生 |
| `synthetic-text` | 随机 id → decode → encode 收敛循环 | 近似（≈ 目标长度） | 两者都支持 |
| `sharegpt` | ShareGPT JSON 真实文本 | **严格精确**（走 ids）/ 近似（走 chat） | 两者都支持 |

### 4.1 `synthetic-ids` 模式

**生成算法**（[`SyntheticIdsGenerator`](bench_multi_turn.py)）：
```python
idx = rng_np.integers(0, len(vocab_arr), size=target_len)
return vocab_arr[idx].tolist()
```
- 从 tokenizer 词表（**排除 special tokens**：pad/bos/eos/added）里等概率抽 `target_len` 个 id
- 直接作为 `List[int]` 传下游，**不经过** decode / encode
- 服务端用 `{"input_ids": [...]}` 字段接收，跳过 tokenize 步骤
- **长度保证**：服务端收到的 prompt token 数 = `sys_len + t*question_len + (t-1)*output_len`，零偏差

**适用场景**：
- 追求"精确控制输入 token 数"的压测
- 不关心文本内容真实性（内容就是随机词表 id decode 出的乱码，仅用于填长度）
- 只能配合 `--backend sglang`（OAI chat API 没有 `input_ids` 字段）

### 4.2 `synthetic-text` 模式

**生成算法**（[`SyntheticTextGenerator`](bench_multi_turn.py)，伪代码）：

```text
1. 先超额生成 ids:  random.choices(vocab, k=int(target_len * 1.2))
2. decode 成 text
3. 进入收敛循环（最多 8 轮）:
     a. ids2 = tokenizer.encode(text, add_special_tokens=False)
     b. if len(ids2) >= target_len: return ids2[:target_len]
     c. else: text += decode(random.choices(...))
4. 兜底: 若仍不足，补 random ids 凑够
```

**为什么需要收敛循环**：BPE tokenizer 的 decode-encode 不守恒，随机 ids decode 的"乱码"再 encode 通常膨胀（2-6x）。通过把 text 用 encode 结果的前 N 个 id 再 decode 一次，就能得到"自身 token 数 == target_len"的稳定文本。

**适用场景**：
- 需要 chat API 链路（服务端必须 tokenize）
- 可以接受 chat template 固定开销（~20 token/轮）

### 4.3 `sharegpt` 模式

**数据源**：`--sharegpt-path` 指向一个 ShareGPT V3 风格的 JSON（结构示意）：

```text
[
  {
    "id": "xxx",
    "conversations": [
      {"from": "human", "value": "用户说的话"},
      {"from": "gpt",   "value": "模型回答"}
    ]
  },
  ...
]
```

**启动时建三个文本池**（[`ShareGPTGenerator._load_pools`](bench_multi_turn.py)）：

| 池名 | 内容 | 用途 |
|---|---|---|
| `prefix` | 所有 `value`（human + gpt） | 填充 system_prompt（只要长内容，什么都行） |
| `question` | 只含 `from=="human"` 的 `value` | 填充每轮用户 question |
| `answer` | 只含 `from=="gpt"` 的 `value` | 填充每轮 A_t 占位 |

**生成算法**（tile 到精确长度）：
```python
buf = []
while len(buf) < target_len:
    text = random.choice(pools[pool_name])
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids:
        buf.extend(ids)
return buf[:target_len]
```
循环从池里抽真实文本、encode 进 buffer，直到攒够 target_len 再精确截断。

**采样策略**（内容差异化）：

| 对象 | 采样次数 | 池 |
|---|---|---|
| `system_prompt[group_id]` | **每 group 采一次**，同 group 所有 conversation 共享 | `prefix` |
| `question[group_id][conv_id][turn_idx]` | 每个位置**独立采样** | `question` |
| `answer_placeholder[group_id][conv_id][turn_idx]` | 每个位置**独立采样** | `answer` |

这样天然达到：
- 同 group 前缀完全相同 → prefix cache 能正常命中 system 部分
- 不同 conversation / 不同轮的 question 和 answer 内容都不同 → 后半段不会被 cache "假命中"

**适用场景**：
- 想让内容贴近真实对话分布（真实 user 风格问题 + 真实 gpt 风格回答占位）
- 既要**精确长度**又要**内容真实**（走 `--backend sglang` + `--dataset-mode sharegpt`）

## 5. 后端与请求协议

用 `--backend` 切换：

| backend | URL | payload 字段 | 精度 | 支持模式 |
|---|---|---|---|---|
| `sglang`（默认） | `/generate` | `input_ids`（ids 模式）或 `text`（text 模式） | **精确**（ids 路径） | 全部三种 |
| `sglang-oai-chat` | `/v1/chat/completions` | `messages`（结构化对话） | 近似（chat template 加 ~20 token） | `synthetic-text`, `sharegpt` |

### 5.1 `sglang` 原生 `/generate` 的 payload

对于每一轮请求，构造如下 payload（示意，`input_ids` 是真实的整数列表）：

```text
POST /generate
{
  "input_ids": [ ... system_ids + Q1_ids + A1_placeholder + ... + Qt_ids ... ],
  "sampling_params": {
    "temperature": 0.0,
    "max_new_tokens": <output_len>,
    "ignore_eos": true
  },
  "stream": true,
  "return_logprob": false
}
```

- 多轮的历史**在客户端拼接成一个扁平的 id 序列**再发
- 每轮仍是**独立的一次 HTTP 请求**（不是一次发全部 N 轮）

### 5.2 `sglang-oai-chat` 的 payload

```text
POST /v1/chat/completions
{
  "model": "xxx",
  "messages": [
    {"role": "system",    "content": "<system prompt text>"},
    {"role": "user",      "content": "<Q1 text>"},
    {"role": "assistant", "content": "<A1 placeholder text>"},
    {"role": "user",      "content": "<Q2 text>"},
    {"role": "assistant", "content": "<A2 placeholder text>"},
    {"role": "user",      "content": "<Qt text>"}
  ],
  "temperature": 0.0,
  "max_tokens": <output_len>,
  "stream": true,
  "ignore_eos": true,
  "stream_options": {"include_usage": true}
}
```

### 5.3 Routing key

每个 conversation 携带 `X-SMG-Routing-Key: runN_timestamp_gX` header，路由层可以据此做用户亲和调度（同 group 的请求落到同一 worker，最大化 prefix cache 命中）。

## 6. 多轮对话语义

### 6.1 两种模式（A = 默认，B 需加 flag）

| 模式 | 如何生成下一轮的 A_t | 时序 |
|---|---|---|
| **选项 A**（默认） | 启动前**预生成**的占位 ids/text | 每轮串行，A_t 不依赖真实响应 |
| **选项 B**（`--replay-real-response`） | 上一轮服务端**真实返回**的 ids/text | 每轮串行，必须等真实响应 |

### 6.2 关键澄清：两种模式都**逐轮发请求**，不是一次性全发

很多人误以为"预生成占位 = 一次发完 N 轮"。**不是**。两种模式的网络流都是：

```
第 1 轮 POST /generate
  input_ids: [system] + [Q1]
  ← 等服务端响应完成
第 2 轮 POST /generate
  input_ids: [system] + [Q1] + [A1_占位或真实] + [Q2]
  ← 等服务端响应完成
...
第 t 轮 POST /generate
  input_ids: [system] + [Q1] + [A1] + ... + [Qt]
```

**选项 A 只改一件事**：第 t+1 轮拼接时，`A_t` 位置用**预生成的占位**而不是**上一轮响应**。时序和请求数完全一样。

### 6.3 为什么默认用选项 A？

- **长度可预测**：prompt 长度严格 = `sys_len + t*question_len + (t-1)*output_len`，不受模型 EOS 行为影响
- **与压测目标对齐**：压测关心的是吞吐和延迟，A_t 的**语义内容**不重要，但**长度**很重要
- **支持批量预生成**：所有素材在 benchmark 开始前就全部生成好，跑时零额外开销

### 6.4 选项 B 何时用？

- 你想完全模拟真实用户（上一轮模型说什么，下一轮就把它拼回去）
- 你接受：服务端响应长度可能受 EOS / 温度影响，导致 prompt 长度不严格精确
- **限制**：native ids 模式下，需要 server 在 `meta_info` 里返回 `output_ids`；若 server 不返回，会 fallback 到占位（非流式下脚本会带 `return_token_ids=true` 尝试请求）

### 6.5 每轮 prompt 的内容

在 `_run_conversation` 里维护三种历史（按 backend 选一种使用）：

- **native ids**：`history_ids: List[int]`，每轮追加 `turn.question_ids`，响应后追加 `turn.answer_ids`（或真实 `response_ids`）
- **native text**：`history_text_parts: List[str]`，`"".join(parts)` 作为 prompt
- **chat**：`history_messages: List[Dict]`，逐轮追加 `{"role":"user"}` + `{"role":"assistant"}`

## 7. 并发与调度

### 7.1 信号量粒度：**conversation 层**

```python
async def _one_conv(conv):
    async with semaphore:                  # ← 占住信号量
        await _run_conversation(conv, ...) # ← 内部跑完所有 N 轮才返回
```

**含义**：
- 同一时刻最多有 `max_concurrency` 个 conversation 在跑
- 每个 conversation **独占一个信号量槽直到它的 N 轮全部结束**
- **不会**出现"用户 A 跑到第 3 轮就切去跑用户 B"的交错情况

举例：`--num-groups 10 --prompts-per-group 1 --num-turns 10 --max-concurrency 1`：
```
[user X : turn 0 → 1 → ... → 9]    全部 10 轮跑完
[user Y : turn 0 → 1 → ... → 9]    下一个用户开始
...
```

（`X`, `Y` 具体是 group_id 几取决于 `--ordered` 开关，默认 shuffled）

### 7.2 到达速率：Poisson 间隔

```python
conv_rate = request_rate / num_turns
interval = np.random.exponential(1.0 / conv_rate)
await asyncio.sleep(interval)
```

- `--request-rate` 是目标的**总请求速率**（per second）
- 因为每个 conversation 是 `num_turns` 次请求，所以 conversation 到达速率 = `request_rate / num_turns`
- 每个 conversation **进入系统**的间隔服从 Poisson 分布

**注意**：如果实际处理速度跟不上（比如 `max_concurrency` 很小），到达的 conversation 会在信号量前排队，实际吞吐由 `max_concurrency` 和单轮延迟决定，`--request-rate` 可能形同虚设。

### 7.3 顺序 vs 乱序

- 默认 `--ordered=False`：启动时 `random.shuffle(conversations)`，打乱 group 顺序避免同 group 集中到达（防止 prefix cache 命中率被虚高）
- `--ordered=True`：严格按 `(group_id, conv_id)` 字典序发送

## 8. 长度控制

### 8.1 `--length-range-ratio`

默认 `1.0`（全部固定长度），每条请求的 `system_prompt_len` / `question_len` / `output_len` 严格等于 CLI 传入值。

若设为 `<1.0`，每条请求的长度在 `[ratio*L, L]` 范围内均匀采样：
```python
lo = max(1, int(base_len * range_ratio))
length = rng.integers(lo, base_len + 1)
```

**和 `bench_serving.py` 的差异**：原生默认是 `0.0`（在 `[1, L]` 均匀采样，平均值 ≈ L/2），本脚本默认 `1.0`，避免该坑。

### 8.2 长度精度保证

| 数据模式 | 配 `sglang` 原生 | 配 `sglang-oai-chat` |
|---|---|---|
| `synthetic-ids` | **严格精确** | 禁用（CLI 报错） |
| `synthetic-text` | 近似（text encode 接近 target） | 近似 + chat 模板开销 |
| `sharegpt` | **严格精确**（走 input_ids） | 近似 + chat 模板开销 |

- **严格精确**：服务端报告的 `prompt_tokens == prompt_len_expected`
- **近似**：通常 ± 少量（< 5 token），chat 链路下还有 +20 左右的固定开销

## 9. CLI 参数完整说明

### 9.1 拓扑参数

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--num-groups` | int | ✓ | - | 模拟用户数。同 group 共享一段 system_prompt |
| `--prompts-per-group` | int | ✓ | - | 每用户的 conversation 数（独立问题序列数） |
| `--num-turns` | int | - | 1 | 每 conversation 轮数 |

### 9.2 长度参数

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--system-prompt-len` | int | ✓ | - | system 前缀长度（token） |
| `--question-len` | int | ✓ | - | 每轮用户 question 长度 |
| `--output-len` | int | ✓ | - | 每轮 `max_new_tokens` 和占位 A_t 长度 |
| `--length-range-ratio` | float | - | 1.0 | 1.0 固定；<1.0 在 [ratio*L, L] 均匀采样 |

### 9.3 数据生成参数

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--dataset-mode` | choice | - | `synthetic-ids` | `synthetic-ids` / `synthetic-text` / `sharegpt` |
| `--sharegpt-path` | str | 条件必填 | - | 仅 `dataset-mode=sharegpt` 必填；ShareGPT V3 JSON 路径 |
| `--seed` | int | - | 1 | 控制所有随机性（采样、shuffle、Poisson 间隔） |

### 9.4 模型 & tokenizer

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--model` | str | ✓ | - | 模型路径（写入 OAI chat payload 的 `model` 字段，也默认作 tokenizer） |
| `--tokenizer` | str | - | = model | tokenizer 路径（走 HuggingFace `AutoTokenizer`） |

### 9.5 后端 & 网络

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--backend` | choice | - | `sglang` | `sglang` / `sglang-oai-chat` |
| `--host` | str | - | `127.0.0.1` | 服务地址 |
| `--port` | int | - | `30000` | 服务端口 |
| `--api-key` | str | - | - | 写入 `Authorization: Bearer ...` header；未设则读 `$OPENAI_API_KEY` / `$API_KEY` |
| `--header` | list[str] | - | `[]` | 附加 HTTP header，格式 `Key=Value`，可重复 |

### 9.6 并发 & 速率

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--num-prompts` | int | - | `groups*prompts*turns` | 总请求数上限；超过时截断 conversation 的尾部 turn |
| `--max-concurrency` | int | - | 16 | 同时运行的 conversation 数（粒度在 conv 层） |
| `--request-rate` | float | - | `inf` | 目标总请求速率（req/s）；`inf` = 不限 |
| `--ordered` | flag | - | False | 开启后按 group/conv 顺序发送；默认 shuffle |

### 9.7 多轮行为

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--replay-real-response` | flag | - | False | 切到选项 B：用服务端真实响应拼历史 |

### 9.8 输出

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--output-file` | str | - | - | per-request 指标 jsonl（不写即不生成） |
| `--summary-csv` | str | - | - | **追加**一行聚合结果到 CSV（首次会写表头） |
| `--dump-prompts-dir` | str | - | - | 写 `content.jsonl` + `requests.jsonl` 用于精度审计 |
| `--case-name` | str | - | `unnamed` | 写入 summary CSV 的 `case_name` 列 |

### 9.9 其他

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `--disable-stream` | flag | - | False | 关闭流式响应；TTFT 将退化为完整响应延迟 |
| `--disable-ignore-eos` | flag | - | False | 允许模型提前 EOS；实际 `output_len` 可能小于目标值 |

## 10. 输出文件

三个输出**全部默认关闭**，各自独立开关。

### 10.1 `--output-file PATH`（per-request JSONL）

**一行 = 一次 HTTP 请求**（一个三元组 `(group_id, conv_id, turn_idx)`）：

```json
{
  "group_id": 0,
  "conv_id": 3,
  "turn_idx": 2,
  "prompt_len_expected": 16512,
  "prompt_len_actual": 16512,
  "output_len_expected": 256,
  "output_len_actual": 256,
  "start_ts": 1714970000.123,
  "first_token_ts": 1714970000.456,
  "end_ts": 1714970002.789,
  "ttft_ms": 333.0,
  "tpot_ms": 9.1,
  "e2e_ms": 2666.0,
  "success": true,
  "error": null,
  "prompt_ids_preview": null,
  "prompt_text_preview": null
}
```

**用途**：画 latency 分布、按 `turn_idx` 分层分析 TTFT（第 1 轮 vs 第 N 轮的差异能直观看出 prefix cache 效果）、定位慢请求。

### 10.2 `--summary-csv PATH`（聚合 CSV）

**一行 = 本次 benchmark 的汇总**。首次写入会创建表头；后续同路径会**追加**，便于多轮参数扫表。

**字段分类**（含义见 [第 11 节](#11-指标与汇总字段)）：

| 分类 | 字段 |
|---|---|
| 标识 | `case_name`, `case_start_time` |
| 拓扑参数 | `num_groups`, `prompts_per_group`, `system_prompt_len`, `question_len`, `output_len`, `num_turns`, `num_prompts` |
| 调度参数 | `request_rate`, `max_concurrency` |
| 结果统计 | `completed`, `failed`, `duration_s` |
| 吞吐量 | `request_throughput`, `input_throughput`, `output_throughput` |
| TTFT 延迟 | `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms` |
| TPOT 延迟 | `mean_tpot_ms`, `p99_tpot_ms` |
| E2E 延迟 | `mean_e2e_latency_ms`, `p90_e2e_latency_ms`, `p99_e2e_latency_ms` |
| 长度精度 | `mean_prompt_len_expected`, `mean_prompt_len_actual`, `mean_prompt_len_diff` |

**配合扫表脚本使用**：
```bash
for conc in 2 4 8 16 32; do
  python -m sglang.benchmark.bench_multi_turn \
    --case-name "conc${conc}" \
    --max-concurrency $conc \
    --summary-csv all_runs.csv \
    ...
done
```
然后直接把 `all_runs.csv` 拉到 Excel 就是完整对比表。

### 10.3 `--dump-prompts-dir DIR`（精度审计双文件）

打开后，在 DIR 下产出两个 JSONL：

#### `content.jsonl` —— 生成素材审计

**一行 = 一个 conversation**，包含 system_prompt + 所有轮次的 question / answer_placeholder。示例（结构示意）：

```text
{
  "group_id": 0,
  "conv_id": 3,
  "routing_key": "run1_<timestamp>_g0",
  "system_prompt": {
    "expected_len": 14400,
    "actual_len":   14400,
    "ids_preview":  [12, 45, 67, ...],
    "text_preview": null
  },
  "turns": [
    {
      "turn_idx": 0,
      "question":           {"expected_len": 1600, "actual_len": 1600, "ids_preview": [...], "text_preview": null},
      "answer_placeholder": {"expected_len": 256,  "actual_len": 256,  "ids_preview": [...], "text_preview": null}
    },
    {
      "turn_idx": 1,
      "question":           {"expected_len": 1600, "actual_len": 1600, "ids_preview": [...], "text_preview": null},
      "answer_placeholder": {"expected_len": 256,  "actual_len": 256,  "ids_preview": [...], "text_preview": null}
    }
  ]
}
```

检查点：`actual_len` 应当与 `expected_len` 完全一致（synthetic-ids / sharegpt 严格相等；synthetic-text 可能有 ±1-2 的 BPE 漂移）。

#### `requests.jsonl` —— 每次 HTTP 请求审计

**一行 = 一次实际发出的请求**，含 client 期望长度 + 服务端 reported 长度 + 预览。字段说明：

| 字段 | 含义 |
|---|---|
| `group_id`, `conv_id`, `turn_idx` | 定位该请求在三级结构中的位置 |
| `prompt_len_expected` | 客户端组装时期望的 prompt token 数 |
| `prompt_len_actual` | 服务端 `meta_info.prompt_tokens` / `usage.prompt_tokens` |
| `output_len_expected` | `max_new_tokens` 设定值 |
| `output_len_actual` | 服务端 `meta_info.completion_tokens` |
| `start_ts`, `first_token_ts`, `end_ts` | 请求各关键时间点（`time.perf_counter()`） |
| `ttft_ms`, `tpot_ms`, `e2e_ms` | 三大延迟指标 |
| `success`, `error` | 请求是否成功及错误信息 |
| `prompt_ids_preview` | native ids 模式下，prompt 前 50 个 id |
| `prompt_text_preview` | text / chat 模式下，prompt 前 200 字符 |

**精度验证**：`prompt_len_actual - prompt_len_expected` 应当：

- `== 0`（native + synthetic-ids / sharegpt）
- `≈ 0`（native + synthetic-text，一般 ±1-2）
- `≈ fixed_chat_overhead`（oai-chat，通常 +10~30 的固定偏移）

## 11. 指标与汇总字段

### 11.1 per-request 指标定义

| 字段 | 含义 |
|---|---|
| `ttft_ms` | Time To First Token：发送请求到收到第一个生成 token 的时间 |
| `tpot_ms` | Time Per Output Token：`(end_ts - first_token_ts) / (output_len - 1)` |
| `e2e_ms` | End-to-End：发送请求到响应完成的总时间 |
| `prompt_len_actual` | 服务端 `meta_info.prompt_tokens`（native）或 `usage.prompt_tokens`（OAI） |
| `output_len_actual` | 服务端 `meta_info.completion_tokens` |

### 11.2 summary 聚合字段

**吞吐量**（只统计成功请求）：
- `request_throughput = completed / duration_s`
- `input_throughput = sum(prompt_len_actual) / duration_s`
- `output_throughput = sum(output_len_actual) / duration_s`

**延迟**（只统计成功请求）：
- TTFT：`mean`, `median`, `p99`
- TPOT：`mean`, `p99`
- E2E：`mean`, `p90`, `p99`

**精度**（所有成功请求的长度期望 vs 实测）：
- `mean_prompt_len_expected`
- `mean_prompt_len_actual`
- `mean_prompt_len_diff = actual - expected`

## 12. 精度验证工作流

第一次用本脚本时，推荐按以下步骤验证长度精度是否达预期：

### 步骤 1：跑一个小规模测试 + dump

```bash
python -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model /path/to/model --tokenizer /path/to/tokenizer \
  --dataset-mode synthetic-ids \
  --num-groups 1 --prompts-per-group 2 --num-turns 3 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 1 --request-rate inf \
  --dump-prompts-dir /tmp/audit \
  --case-name precision_probe
```

### 步骤 2：查 `content.jsonl`

```bash
cat /tmp/audit/content.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    row = json.loads(line)
    sp = row['system_prompt']
    print(f'g{row[\"group_id\"]}/c{row[\"conv_id\"]} system: exp={sp[\"expected_len\"]} act={sp[\"actual_len\"]}')
    for t in row['turns']:
        q, a = t['question'], t['answer_placeholder']
        print(f'  turn{t[\"turn_idx\"]}: Q exp={q[\"expected_len\"]} act={q[\"actual_len\"]}; A exp={a[\"expected_len\"]} act={a[\"actual_len\"]}')
"
```

**期望**：所有 `actual_len == expected_len`（synthetic-ids 模式）。

### 步骤 3：查 `requests.jsonl`

对照 client 期望和 server reported：
```bash
cat /tmp/audit/requests.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    diff = r['prompt_len_actual'] - r['prompt_len_expected']
    print(f'g{r[\"group_id\"]}/c{r[\"conv_id\"]}/t{r[\"turn_idx\"]}: exp={r[\"prompt_len_expected\"]} act={r[\"prompt_len_actual\"]} diff={diff:+d}')
"
```

**期望**：每条 diff 为 0。若不为 0，根据 [第 10.3 节](#10-输出文件) 的预期偏移表判断是否符合模式。

### 步骤 4：确认多轮累积正确

同一 `(group_id, conv_id)` 的不同 `turn_idx`，`prompt_len_actual` 应当严格递增且增量符合预期：
- 第 1 轮 → 14400 + 1600 = 16000
- 第 2 轮 → 16000 + 256 + 1600 = 17856
- 第 3 轮 → 17856 + 256 + 1600 = 19712

若出现：
- 所有轮都是 16000 → 历史没累积，是 bug
- 不规律增长 → 检查 `--length-range-ratio` 是不是 <1

## 13. 兼容性与依赖

### 13.1 SGLang 源码依赖

**本脚本只从 sglang 源码 import 一样东西**：

```python
from sglang.benchmark.utils import get_tokenizer
```

[utils.py:44](utils.py#L44) 里的 `get_tokenizer` 就是 HuggingFace `AutoTokenizer.from_pretrained` 的薄包装，历史上接口稳定，跨版本兼容性好。

### 13.2 第三方依赖

- `aiohttp`
- `numpy`
- `transformers`（由 `get_tokenizer` 间接依赖）
- `tqdm`

这些都是 SGLang 的标配依赖，在 SGLang 运行环境里默认可用。

### 13.3 服务端 API 兼容

| 后端 | 要求 |
|---|---|
| `sglang` | SGLang server 的 `/generate` 必须接受 `input_ids` 字段（参见 [`async_request_sglang_generate`](../bench_serving.py)）。所有主线版本均支持 |
| `sglang-oai-chat` | SGLang server 的 `/v1/chat/completions` 支持 `ignore_eos`, `stream_options.include_usage` |

`--replay-real-response` 下的 native ids 模式，依赖 server 在 `meta_info` 里返回 `output_ids`。若 server 不返回，会自动 fallback 到占位 A_t。

### 13.4 路由层

header `X-SMG-Routing-Key` 的取值是 `f"run{seed}_{timestamp}_g{group_id}"`，同 group 的所有请求 key 相同。若你的路由层用其他 header 名，可用 `--header` 覆盖或扩展（但当前脚本仍会写入 `X-SMG-Routing-Key`）。

## 14. 已知限制

1. **轮内只支持串行**：同一 conversation 的 N 轮强制 await 响应完才能发下一轮，不支持同 conv 内并发（设计如此，贴近真实用户节奏）
2. **多轮交错不支持**：信号量在 conversation 粒度，无法做"N 个用户轮流"的交错调度。若需要，可改 `_one_conv` 把信号量下沉到每个 `await send`
3. **`--request-rate` 在低并发下可能失效**：若 `max_concurrency` 明显小于 `request_rate * num_turns`，实际速率由并发瓶颈决定
4. **chat 模式无法严格控长**：chat template 在服务端生效，客户端无法预知确切开销，只能接受近似
5. **`synthetic-text` 在某些 tokenizer 上可能不收敛**：罕见情况下 BPE 合并诡异，收敛循环会触发兜底（拼足长度但不保证精确）；遇到可切 `synthetic-ids`
6. **`--disable-ignore-eos` + `--replay-real-response` 下 prompt 长度不稳**：模型提前 EOS 会导致 `response_ids` 短于 `output_len`，history 长度不再可预测

## 15. 常见使用示例

### 15.1 复现你 GSP 原有的多轮场景（严格精确版）

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 38998 \
  --model /gpfs/models/huggingface.co/deepseek-ai/DeepSeek-V3.2 \
  --tokenizer /gpfs/models/huggingface.co/deepseek-ai/DeepSeek-V3.2 \
  --dataset-mode synthetic-ids \
  --num-groups 1 --prompts-per-group 40 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 16 --request-rate 16 \
  --summary-csv bench_results/multi_turn.csv \
  --case-name gsp_repro_synth
```

### 15.2 使用 ShareGPT 真实内容

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 38998 \
  --model /gpfs/models/huggingface.co/deepseek-ai/DeepSeek-V3.2 \
  --tokenizer /gpfs/models/huggingface.co/deepseek-ai/DeepSeek-V3.2 \
  --dataset-mode sharegpt \
  --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-groups 10 --prompts-per-group 4 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 16 --request-rate 16 \
  --summary-csv bench_results/multi_turn.csv \
  --case-name sharegpt_conc16
```

### 15.3 并发扫表（写入同一个 CSV 连续追加）

```bash
for conc in 2 4 8 16 32 64; do
  python3 -m sglang.benchmark.bench_multi_turn \
    --backend sglang --host 127.0.0.1 --port 38998 \
    --model /path/to/model --tokenizer /path/to/model \
    --dataset-mode sharegpt \
    --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-groups $conc --prompts-per-group 1 --num-turns 10 \
    --system-prompt-len 14400 --question-len 1600 --output-len 256 \
    --max-concurrency $conc --request-rate $conc \
    --summary-csv bench_results/concurrency_sweep.csv \
    --case-name "sharegpt_conc${conc}"
done
```

### 15.4 精度校验小规模跑

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 38998 \
  --model /path/to/model --tokenizer /path/to/model \
  --dataset-mode synthetic-ids \
  --num-groups 2 --prompts-per-group 2 --num-turns 3 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 1 --request-rate inf \
  --dump-prompts-dir bench_results/audit \
  --case-name audit_synth_ids
```

### 15.5 chat API 链路测试

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang-oai-chat --host 127.0.0.1 --port 38998 \
  --model /path/to/model --tokenizer /path/to/model \
  --dataset-mode sharegpt \
  --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-groups 4 --prompts-per-group 4 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 16 --request-rate 16
```

### 15.6 严格真实响应回填（选项 B）

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 38998 \
  --model /path/to/model --tokenizer /path/to/model \
  --dataset-mode synthetic-ids \
  --num-groups 1 --prompts-per-group 4 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 4 --request-rate inf \
  --replay-real-response \
  --disable-stream \
  --dump-prompts-dir bench_results/real_response_audit
```
（`--disable-stream` 让 server 在非流式响应里一次性返回 `output_ids`，提高 response_ids 回填成功率）

### 15.7 单用户严格串行（排查单请求延迟）

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 38998 \
  --model /path/to/model --tokenizer /path/to/model \
  --dataset-mode synthetic-ids \
  --num-groups 1 --prompts-per-group 1 --num-turns 10 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 1 --request-rate inf \
  --ordered \
  --output-file bench_results/single_user.jsonl
```

---

## 附录：关键源码位置快查

| 功能 | 位置 |
|---|---|
| Config / dataclass 定义 | `bench_multi_turn.py` 开头 `@dataclass` 段 |
| CLI argparse | `build_argparser`, `args_to_config`, `validate_config` |
| 三种数据生成器 | `SyntheticIdsGenerator` / `SyntheticTextGenerator` / `ShareGPTGenerator` |
| 预生成 conversations | `generate_conversations` |
| native /generate 发送器 | `_send_native_generate` |
| chat /v1/chat/completions 发送器 | `_send_oai_chat` |
| 每 conversation 串行多轮 | `_run_conversation` |
| 并发调度 + 到达速率 | `_conversation_arrival`, `run_benchmark` |
| 汇总聚合 | `_compute_summary`, `Summary` dataclass |
| 三种输出 | `_write_per_request_jsonl`, `_append_summary_csv`, `_write_content_dump` |
| 入口 | `main` |

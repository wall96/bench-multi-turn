# bench-multi-turn

一个基于 [SGLang](https://github.com/sgl-project/sglang) 的多轮对话基准测试脚本。

相比 `sglang.bench_serving` 的 `generated-shared-prefix` / `random` / `random-ids` 数据集，本脚本：

- **严格可控的 prompt 长度**：走 `input_ids` 直通通道，绕过 tokenizer 的 decode/encode 往返不守恒问题（原生脚本常见 2–6× 长度膨胀）
- **贴近真实用户的多轮对话语义**：同一用户（group）共享长前缀 system_prompt，conversation 内逐轮累积历史，每轮是独立的 HTTP 请求
- **三种数据来源**：纯随机 token ids / 随机 id 收敛到文本 / ShareGPT 真实对话采样
- **细粒度精度审计**：可选 dump 完整的生成素材 + 每次请求 payload + 服务端 reported 长度
- **数据集可复用**：支持把数据集 dump 下来下次直接加载，或编辑 text 后重新 tokenize 再跑

## 快速开始

### 前置

把脚本放到 SGLang 源码树的 `python/sglang/benchmark/` 下：

```bash
cp bench_multi_turn.py <sglang-root>/python/sglang/benchmark/
```

依赖（SGLang 运行环境里默认可用）：`aiohttp`, `numpy`, `transformers`, `tqdm`。

### 最简示例：随机 ids 精确控长

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model /path/to/model --tokenizer /path/to/tokenizer \
  --dataset-mode synthetic-ids \
  --num-groups 1 --prompts-per-group 40 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 16 --request-rate 16
```

### ShareGPT 真实内容

```bash
python3 -m sglang.benchmark.bench_multi_turn \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model /path/to/model --tokenizer /path/to/tokenizer \
  --dataset-mode sharegpt \
  --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-groups 10 --prompts-per-group 4 --num-turns 5 \
  --system-prompt-len 14400 --question-len 1600 --output-len 256 \
  --max-concurrency 16 --request-rate 16 \
  --summary-csv results.csv --case-name conc16
```

### 复用上次的数据集

```bash
# 第一次：生成并保存完整数据
python3 -m sglang.benchmark.bench_multi_turn ... \
  --dump-prompts-dir my_dataset --dump-full-content

# 后续：直接加载
python3 -m sglang.benchmark.bench_multi_turn ... \
  --load-dataset my_dataset/content.jsonl \
  --max-concurrency 32  # 并发可自由改
```

## 核心概念速览

```
Group（用户） = 一个 system_prompt（只采样一次，同 group 共享）
 └── Conversation —— 独立的问答序列
      └── Turn —— 每轮是一次独立 HTTP 请求
           prompt = system + Q1 + A1_占位 + Q2 + A2_占位 + ... + Qt
```

- **默认选项 A**：A_t 用预生成的占位 ids（长度可预测，不依赖服务端真实响应）
- **选项 B**（`--replay-real-response`）：用服务端真实响应拼历史

## 完整文档

详细的参数说明、设计细节、精度验证工作流、常见使用场景，见 [`bench_multi_turn.md`](bench_multi_turn.md)。

## 三种数据模式对比

| 模式 | 数据来源 | 长度精度 | 支持后端 |
|---|---|---|---|
| `synthetic-ids`（默认） | 随机采样词表 id | **严格精确** | 仅 `sglang` 原生 |
| `synthetic-text` | 随机 id → decode → encode 收敛循环 | 近似（≈ 目标长度） | 两者都支持 |
| `sharegpt` | ShareGPT JSON 真实文本 | **严格精确**（走 ids）/ 近似（走 chat） | 两者都支持 |

## 主要 CLI 参数

| 类别 | 参数 |
|---|---|
| 拓扑 | `--num-groups` / `--prompts-per-group` / `--num-turns` |
| 长度 | `--system-prompt-len` / `--question-len` / `--output-len` / `--length-range-ratio` |
| 数据 | `--dataset-mode` / `--sharegpt-path` / `--seed` |
| 数据集复用 | `--load-dataset` / `--load-from {ids,text}` |
| 后端 | `--backend` / `--host` / `--port` / `--model` / `--tokenizer` |
| 并发 | `--max-concurrency` / `--request-rate` / `--num-prompts` / `--ordered` |
| 多轮 | `--replay-real-response` |
| 输出 | `--output-file` / `--summary-csv` / `--dump-prompts-dir` / `--dump-full-content` |

完整参数表见 [文档第 9 节](bench_multi_turn.md)。

## License

MIT（如需其他许可证，请修改本节并添加对应 LICENSE 文件）。

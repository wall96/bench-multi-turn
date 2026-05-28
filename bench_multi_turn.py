"""Multi-turn conversation benchmark for SGLang.

Simulates a set of users (groups), each issuing several multi-turn conversations
against an SGLang server. Each user shares a fixed-length system prompt; within a
conversation the client sends turns sequentially, each turn's prompt being
(system + Q1 + A1_placeholder + ... + Q_t) where A_i are pre-generated
placeholders by default (ignored from the real server response).

Three data generation modes:
  synthetic-ids   : pure random token ids, bypass tokenizer (exact length,
                    sglang native backend only)
  synthetic-text  : random ids -> decode -> encode-adjust loop
  sharegpt        : sample real text from ShareGPT JSON, tile to exact length

Usage:
  python -m sglang.benchmark.bench_multi_turn \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --model /path/to/model --tokenizer /path/to/tokenizer \
    --dataset-mode sharegpt --sharegpt-path /path/to/sharegpt.json \
    --num-groups 4 --prompts-per-group 10 --num-turns 5 \
    --system-prompt-len 14400 --question-len 1600 --output-len 256 \
    --max-concurrency 16 --request-rate 16
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer


def get_tokenizer(pretrained_model_name_or_path: str):
    """Load a HF tokenizer. Kept as a tiny wrapper so the script runs
    standalone without any sglang-internal imports."""
    assert pretrained_model_name_or_path, "tokenizer path is required"
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATASET_MODES = ("synthetic-ids", "synthetic-text", "sharegpt")
BACKENDS = ("sglang", "sglang-oai-chat")

AIOHTTP_TIMEOUT_SEC = 6 * 60 * 60
AIOHTTP_READ_BUFSIZE = 10 * 1024 * 1024

_SPECIAL_TOKEN_FILTER_ATTRS = (
    "all_special_ids",
    "added_tokens_encoder",
)


# --------------------------------------------------------------------------- #
# Config / data structures
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # topology
    num_groups: int
    prompts_per_group: int
    num_turns: int
    # lengths
    system_prompt_len: int
    question_len: int
    output_len: int
    length_range_ratio: float
    # data
    dataset_mode: str
    sharegpt_path: Optional[str]
    load_dataset: Optional[str]
    load_from: str
    seed: int
    # model
    model: str
    tokenizer: str
    # backend
    backend: str
    host: str
    port: int
    api_key: Optional[str]
    extra_headers: List[str]
    extra_request_body: Dict[str, Any]
    # concurrency
    num_prompts: int
    max_concurrency: int
    request_rate: float
    ordered: bool
    # multi-turn
    replay_real_response: bool
    # outputs
    output_file: Optional[str]
    summary_csv: Optional[str]
    dump_prompts_dir: Optional[str]
    dump_full_content: bool
    # misc
    disable_stream: bool
    disable_ignore_eos: bool
    case_name: Optional[str]

    @property
    def total_requests(self) -> int:
        return self.num_groups * self.prompts_per_group * self.num_turns


@dataclass
class TurnMaterials:
    """Pre-generated content for one (group, conv, turn)."""
    turn_idx: int
    question_ids: List[int]
    answer_ids: List[int]           # placeholder; actual length = output_len
    question_text: Optional[str] = None
    answer_text: Optional[str] = None

    @property
    def question_len(self) -> int:
        return len(self.question_ids)

    @property
    def output_len(self) -> int:
        return len(self.answer_ids)


@dataclass
class Conversation:
    group_id: int
    conv_id: int
    system_ids: List[int]
    turns: List[TurnMaterials]
    system_text: Optional[str] = None
    routing_key: Optional[str] = None

    @property
    def system_len(self) -> int:
        return len(self.system_ids)


@dataclass
class RequestRecord:
    group_id: int
    conv_id: int
    turn_idx: int
    # lengths
    prompt_len_expected: int
    prompt_len_actual: int
    output_len_expected: int
    output_len_actual: int
    # timing
    start_ts: float
    first_token_ts: float
    end_ts: float
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    # status
    success: bool
    error: Optional[str]
    # streaming-derived metrics; default 0/empty so non-streaming and old
    # consumers still work.
    tpot_corrected_ms: float = 0.0
    num_chunks: int = 0
    first_chunk_tokens: int = 0
    itl_mean_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p90_ms: float = 0.0
    itl_p99_ms: float = 0.0
    # preview (for dump)
    prompt_ids_preview: Optional[List[int]] = None
    prompt_text_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sglang.benchmark.bench_multi_turn",
        description="Multi-turn conversation benchmark for SGLang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # topology (required unless --load-dataset is used)
    p.add_argument("--num-groups", type=int, default=None,
                   help="Number of simulated users; each user shares one system "
                        "prompt. Required unless --load-dataset is set.")
    p.add_argument("--prompts-per-group", type=int, default=None,
                   help="Number of conversations per user. Required unless "
                        "--load-dataset is set.")
    p.add_argument("--num-turns", type=int, default=1,
                   help="Number of turns per conversation.")
    # lengths (required unless --load-dataset is used)
    p.add_argument("--system-prompt-len", type=int, default=None)
    p.add_argument("--question-len", type=int, default=None)
    p.add_argument("--output-len", type=int, default=None)
    p.add_argument("--length-range-ratio", type=float, default=1.0,
                   help="1.0 = fixed length; <1.0 samples lengths in [ratio*L, L].")
    # data
    p.add_argument("--dataset-mode", choices=DATASET_MODES, default="synthetic-ids")
    p.add_argument("--sharegpt-path", type=str, default=None,
                   help="Path to ShareGPT JSON; required when dataset-mode=sharegpt.")
    p.add_argument("--load-dataset", type=str, default=None,
                   help="Load conversations from a previously dumped "
                        "content.jsonl (requires it was produced with "
                        "--dump-full-content). When set, data generation and "
                        "topology/length CLI args are bypassed.")
    p.add_argument("--load-from", choices=("ids", "text"), default="ids",
                   help="When loading a dumped dataset: 'ids' (default) uses "
                        "the stored ids verbatim (exact replay, lengths stay "
                        "identical). 'text' re-encodes the text_preview field "
                        "to ids (pick this after you manually edit the text; "
                        "lengths may drift due to BPE).")
    p.add_argument("--seed", type=int, default=1)
    # model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer path; defaults to --model.")
    # backend
    p.add_argument("--backend", choices=BACKENDS, default="sglang")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--header", action="append", default=[],
                   help="Extra HTTP header 'Key=Value'; can repeat.")
    p.add_argument("--extra-request-body", type=str, default=None,
                   help="JSON string merged into every request payload. "
                        "Useful for decode-only / fake-prefill benchmarking, "
                        "e.g. --extra-request-body "
                        "'{\"bootstrap_host\": \"2.2.2.2\", \"bootstrap_room\": 0}' "
                        "when the decode server is launched with "
                        "--disaggregation-transfer-backend fake.")
    # concurrency
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Cap on total requests sent (default: groups*prompts*turns).")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Max requests per second; inf = unlimited.")
    p.add_argument("--ordered", action="store_true",
                   help="Send requests in group order; default is shuffled.")
    # multi-turn
    p.add_argument("--replay-real-response", action="store_true",
                   help="Option B: build next turn's prompt from real model response.")
    # outputs
    p.add_argument("--output-file", type=str, default=None,
                   help="Per-request metrics jsonl.")
    p.add_argument("--summary-csv", type=str, default=None,
                   help="Append one aggregated row to this CSV.")
    p.add_argument("--dump-prompts-dir", type=str, default=None,
                   help="Dump content.jsonl and requests.jsonl for precision audit.")
    p.add_argument("--dump-full-content", action="store_true",
                   help="Dump FULL ids/text in the dump files instead of only "
                        "previews (50 ids / 200 chars). File size grows "
                        "linearly with total tokens; use with care.")
    # misc
    p.add_argument("--disable-stream", action="store_true",
                   help="Disable streaming response (TTFT degrades to full-response latency).")
    p.add_argument("--disable-ignore-eos", action="store_true",
                   help="Allow model to emit EOS early; output_len may be less than requested.")
    p.add_argument("--case-name", type=str, default=None,
                   help="Label written into summary-csv row.")
    return p


def args_to_config(args: argparse.Namespace) -> Config:
    tokenizer = args.tokenizer or args.model
    extra_body: Dict[str, Any] = {}
    if args.extra_request_body:
        try:
            extra_body = json.loads(args.extra_request_body)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--extra-request-body is not valid JSON: {e}"
            ) from None
        if not isinstance(extra_body, dict):
            raise ValueError("--extra-request-body must be a JSON object.")
    if args.load_dataset:
        # Topology/length fields will be set from the loaded data later.
        # Use 0 as placeholders; num_prompts cap is also resolved post-load.
        ng = args.num_groups or 0
        ppg = args.prompts_per_group or 0
        spl = args.system_prompt_len or 0
        ql = args.question_len or 0
        ol = args.output_len or 0
        num_prompts = args.num_prompts if args.num_prompts is not None else 0
    else:
        ng = args.num_groups
        ppg = args.prompts_per_group
        spl = args.system_prompt_len
        ql = args.question_len
        ol = args.output_len
        if ng is None or ppg is None:
            total = 0
        else:
            total = ng * ppg * args.num_turns
        num_prompts = args.num_prompts if args.num_prompts is not None else total
        if total > 0 and num_prompts > total:
            num_prompts = total

    cfg = Config(
        num_groups=ng,
        prompts_per_group=ppg,
        num_turns=args.num_turns,
        system_prompt_len=spl,
        question_len=ql,
        output_len=ol,
        length_range_ratio=args.length_range_ratio,
        dataset_mode=args.dataset_mode,
        sharegpt_path=args.sharegpt_path,
        load_dataset=args.load_dataset,
        load_from=args.load_from,
        seed=args.seed,
        model=args.model,
        tokenizer=tokenizer,
        backend=args.backend,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        extra_headers=list(args.header or []),
        extra_request_body=extra_body,
        num_prompts=num_prompts,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        ordered=args.ordered,
        replay_real_response=args.replay_real_response,
        output_file=args.output_file,
        summary_csv=args.summary_csv,
        dump_prompts_dir=args.dump_prompts_dir,
        dump_full_content=args.dump_full_content,
        disable_stream=args.disable_stream,
        disable_ignore_eos=args.disable_ignore_eos,
        case_name=args.case_name,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    # When loading a pre-generated dataset, topology/length/sharegpt fields are
    # not required -- they are derived from the loaded file.
    if cfg.load_dataset:
        if not (0.0 < cfg.length_range_ratio <= 1.0):
            raise ValueError("--length-range-ratio must be in (0, 1].")
        return

    if cfg.dataset_mode == "sharegpt" and not cfg.sharegpt_path:
        raise ValueError("--sharegpt-path is required when --dataset-mode=sharegpt")
    if cfg.dataset_mode == "synthetic-ids" and cfg.backend == "sglang-oai-chat":
        raise ValueError(
            "--dataset-mode=synthetic-ids is incompatible with "
            "--backend=sglang-oai-chat (chat API has no input_ids field). "
            "Use --dataset-mode=synthetic-text or sharegpt."
        )
    if not (0.0 < cfg.length_range_ratio <= 1.0):
        raise ValueError("--length-range-ratio must be in (0, 1].")

    required = {
        "num-groups": cfg.num_groups,
        "prompts-per-group": cfg.prompts_per_group,
        "system-prompt-len": cfg.system_prompt_len,
        "question-len": cfg.question_len,
        "output-len": cfg.output_len,
    }
    for name, v in required.items():
        if v is None or v <= 0:
            raise ValueError(
                f"--{name} is required (and must be positive) when "
                f"--load-dataset is not set"
            )


def _jitter_lens(base_len: int, range_ratio: float, n: int,
                 rng: np.random.Generator) -> List[int]:
    if range_ratio >= 1.0:
        return [base_len] * n
    lo = max(1, int(base_len * range_ratio))
    return rng.integers(lo, base_len + 1, size=n).tolist()


# --------------------------------------------------------------------------- #
# Tokenizer helpers
# --------------------------------------------------------------------------- #

def _non_special_vocab_ids(tokenizer) -> List[int]:
    """Return vocabulary ids excluding special tokens (pad/bos/eos/added)."""
    vocab = tokenizer.get_vocab()
    all_ids = set(vocab.values())
    special_ids = set()
    for attr in _SPECIAL_TOKEN_FILTER_ATTRS:
        v = getattr(tokenizer, attr, None)
        if v is None:
            continue
        try:
            if isinstance(v, dict):
                special_ids.update(v.values())
            else:
                special_ids.update(v)
        except TypeError:
            pass
    return sorted(all_ids - special_ids)


def _encode_noadd(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


# --------------------------------------------------------------------------- #
# Data generators
# --------------------------------------------------------------------------- #

class BaseGenerator:
    """Base class for prompt/question/answer generation.

    Subclasses implement gen_ids(target_len) -> List[int] and
    gen_text(target_len) -> str. By default gen_text delegates to gen_ids and
    decodes; subclasses can override for precision.
    """

    def __init__(self, tokenizer, cfg: Config, rng_py: random.Random,
                 rng_np: np.random.Generator):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rng_py = rng_py
        self.rng_np = rng_np

    # -- interface ----------------------------------------------------------
    def gen_ids(self, target_len: int, pool: str = "any") -> List[int]:
        raise NotImplementedError

    def gen_text(self, target_len: int, pool: str = "any") -> Tuple[str, List[int]]:
        """Return (text, ids) where len(ids) == target_len exactly.

        The returned text re-tokenizes to approximately target_len ids; the
        authoritative length is whatever `ids` contains.
        """
        ids = self.gen_ids(target_len, pool=pool)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, ids


class SyntheticIdsGenerator(BaseGenerator):
    """Pure random token ids; exact length guaranteed."""

    def __init__(self, tokenizer, cfg, rng_py, rng_np):
        super().__init__(tokenizer, cfg, rng_py, rng_np)
        self._vocab = _non_special_vocab_ids(tokenizer)
        if not self._vocab:
            raise RuntimeError("Tokenizer has no usable (non-special) ids.")
        self._vocab_arr = np.asarray(self._vocab, dtype=np.int64)

    def gen_ids(self, target_len: int, pool: str = "any") -> List[int]:
        if target_len <= 0:
            return []
        idx = self.rng_np.integers(0, len(self._vocab_arr), size=target_len)
        return self._vocab_arr[idx].tolist()


class SyntheticTextGenerator(BaseGenerator):
    """Random ids -> decode -> encode-adjust loop to converge on exact length."""

    MAX_ITER = 8

    def __init__(self, tokenizer, cfg, rng_py, rng_np):
        super().__init__(tokenizer, cfg, rng_py, rng_np)
        self._vocab = _non_special_vocab_ids(tokenizer)

    def _random_ids(self, n: int) -> List[int]:
        return self.rng_py.choices(self._vocab, k=n)

    def gen_ids(self, target_len: int, pool: str = "any") -> List[int]:
        if target_len <= 0:
            return []
        # Start from oversized random ids then converge via decode-encode.
        ids = self._random_ids(max(1, int(target_len * 1.2)))
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        for _ in range(self.MAX_ITER):
            ids2 = _encode_noadd(self.tokenizer, text)
            if len(ids2) >= target_len:
                return ids2[:target_len]
            # too short -> append more random content
            more = self._random_ids(target_len - len(ids2) + 32)
            text = text + self.tokenizer.decode(more, skip_special_tokens=True)
        # Fallback: concat raw, truncate/pad
        ids2 = _encode_noadd(self.tokenizer, text)
        if len(ids2) < target_len:
            ids2 = ids2 + self._random_ids(target_len - len(ids2))
        return ids2[:target_len]

    def gen_text(self, target_len: int, pool: str = "any") -> Tuple[str, List[int]]:
        ids = self.gen_ids(target_len, pool=pool)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, ids


class ShareGPTGenerator(BaseGenerator):
    """Sample real text from ShareGPT JSON; tile tokens to exact length."""

    # Pools:
    #   "prefix"   -> all values (human + gpt)  used for system_prompt
    #   "question" -> "human" role values only   used for user questions
    #   "answer"   -> "gpt"   role values only   used for A_t placeholders
    POOLS = ("prefix", "question", "answer")

    def __init__(self, tokenizer, cfg, rng_py, rng_np):
        super().__init__(tokenizer, cfg, rng_py, rng_np)
        if not cfg.sharegpt_path:
            raise ValueError("sharegpt_path required")
        self._pools: Dict[str, List[str]] = self._load_pools(cfg.sharegpt_path)
        for name in self.POOLS:
            if not self._pools.get(name):
                raise RuntimeError(
                    f"ShareGPT pool '{name}' is empty after loading "
                    f"{cfg.sharegpt_path}; check the file format."
                )

    @staticmethod
    def _load_pools(path: str) -> Dict[str, List[str]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix: List[str] = []
        question: List[str] = []
        answer: List[str] = []
        for entry in data:
            conv = entry.get("conversations") or entry.get("conversation") or []
            for turn in conv:
                role = (turn.get("from") or "").lower()
                val = turn.get("value") or ""
                if not val:
                    continue
                prefix.append(val)
                if role == "human":
                    question.append(val)
                elif role == "gpt":
                    answer.append(val)
        return {"prefix": prefix, "question": question, "answer": answer}

    def _sample_text(self, pool: str) -> str:
        name = pool if pool in self._pools else "prefix"
        return self.rng_py.choice(self._pools[name])

    def gen_ids(self, target_len: int, pool: str = "prefix") -> List[int]:
        if target_len <= 0:
            return []
        buf: List[int] = []
        # Bound the loop to avoid a pathological infinite loop if all samples
        # tokenize to empty for some reason (shouldn't happen in practice).
        safety = 0
        max_safety = target_len * 4 + 64
        while len(buf) < target_len and safety < max_safety:
            safety += 1
            text = self._sample_text(pool)
            ids = _encode_noadd(self.tokenizer, text)
            if not ids:
                continue
            buf.extend(ids)
        if len(buf) < target_len:
            # Extreme fallback: pad with rotating id from the pool tail
            if not buf:
                raise RuntimeError(
                    "ShareGPT pool produced 0 tokens; aborting generation."
                )
            pad_src = buf * ((target_len // len(buf)) + 1)
            buf = buf + pad_src[: target_len - len(buf)]
        return buf[:target_len]

    def gen_text(self, target_len: int, pool: str = "prefix") -> Tuple[str, List[int]]:
        ids = self.gen_ids(target_len, pool=pool)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, ids


def build_generator(cfg: Config, tokenizer) -> BaseGenerator:
    rng_py = random.Random(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)
    if cfg.dataset_mode == "synthetic-ids":
        return SyntheticIdsGenerator(tokenizer, cfg, rng_py, rng_np)
    if cfg.dataset_mode == "synthetic-text":
        return SyntheticTextGenerator(tokenizer, cfg, rng_py, rng_np)
    if cfg.dataset_mode == "sharegpt":
        return ShareGPTGenerator(tokenizer, cfg, rng_py, rng_np)
    raise ValueError(f"Unknown dataset-mode: {cfg.dataset_mode}")


# --------------------------------------------------------------------------- #
# Materialize all conversations up-front
# --------------------------------------------------------------------------- #

def generate_conversations(cfg: Config, tokenizer,
                           gen: BaseGenerator) -> List[Conversation]:
    """Pre-generate every (group, conv, turn) content deterministically."""
    use_text = cfg.backend == "sglang-oai-chat"
    rng_py = random.Random(cfg.seed + 7919)
    rng_np = np.random.default_rng(cfg.seed + 7919)

    # Per-group system prompts (single sample each)
    group_sys: List[Tuple[List[int], Optional[str]]] = []
    sys_lens = _jitter_lens(cfg.system_prompt_len, cfg.length_range_ratio,
                            cfg.num_groups, rng_np)
    for g in range(cfg.num_groups):
        if use_text:
            text, ids = gen.gen_text(sys_lens[g], pool="prefix")
        else:
            ids = gen.gen_ids(sys_lens[g], pool="prefix")
            text = None
        group_sys.append((ids, text))

    # Per-(group, conv, turn) questions and answer placeholders
    q_lens = _jitter_lens(
        cfg.question_len, cfg.length_range_ratio,
        cfg.num_groups * cfg.prompts_per_group * cfg.num_turns, rng_np,
    )
    a_lens = _jitter_lens(
        cfg.output_len, cfg.length_range_ratio,
        cfg.num_groups * cfg.prompts_per_group * cfg.num_turns, rng_np,
    )

    run_tag = f"run{cfg.seed}_{int(time.time())}"
    conversations: List[Conversation] = []
    idx = 0
    for g in range(cfg.num_groups):
        sys_ids, sys_text = group_sys[g]
        routing_key = f"{run_tag}_g{g}"
        for p in range(cfg.prompts_per_group):
            turns: List[TurnMaterials] = []
            for t in range(cfg.num_turns):
                ql = q_lens[idx]
                al = a_lens[idx]
                idx += 1
                if use_text:
                    q_text, q_ids = gen.gen_text(ql, pool="question")
                    a_text, a_ids = gen.gen_text(al, pool="answer")
                else:
                    q_ids = gen.gen_ids(ql, pool="question")
                    a_ids = gen.gen_ids(al, pool="answer")
                    q_text, a_text = None, None
                turns.append(TurnMaterials(
                    turn_idx=t,
                    question_ids=q_ids,
                    answer_ids=a_ids,
                    question_text=q_text,
                    answer_text=a_text,
                ))
            conversations.append(Conversation(
                group_id=g,
                conv_id=p,
                system_ids=sys_ids,
                system_text=sys_text,
                turns=turns,
                routing_key=routing_key,
            ))
    return conversations


# --------------------------------------------------------------------------- #
# Dataset reload (--load-dataset)
# --------------------------------------------------------------------------- #

def load_conversations_from_dump(
    path: str,
    tokenizer=None,
    prefer: str = "ids",
) -> Tuple[List[Conversation], Dict[str, Any]]:
    """Reconstruct conversations from a previous run's content.jsonl.

    Args:
        path: path to content.jsonl
        tokenizer: required when `prefer == "text"` (to re-encode edited text)
        prefer: "ids" (default) uses stored ids as source of truth;
                "text" re-encodes the text_preview field -- use this after
                manually editing the JSON.

    For `prefer="ids"`: the file must have been produced with
    --dump-full-content (stored id list length must match `expected_len`).

    For `prefer="text"`: the text_preview fields must be populated (they are
    when --dump-full-content was used); the returned lengths reflect the
    re-encoded ids, which may differ from the original expected_len.
    """
    if prefer not in ("ids", "text"):
        raise ValueError(f"prefer must be 'ids' or 'text', got {prefer!r}")
    if prefer == "text" and tokenizer is None:
        raise ValueError("tokenizer is required when prefer='text'")

    conversations: List[Conversation] = []
    max_group_id = -1
    convs_per_group: Dict[int, int] = {}
    max_turns = 0
    sys_lens: List[int] = []
    q_lens: List[int] = []
    a_lens: List[int] = []
    has_any_text = False

    def _resolve(stored_ids, stored_text, where: str):
        """Return (ids, text) using the preferred source as authoritative."""
        if prefer == "text":
            if not stored_text:
                raise ValueError(
                    f"{where}: --load-from=text requires text_preview to be "
                    f"populated, but it is empty/null. Re-dump with "
                    f"--dump-full-content, or use --load-from=ids."
                )
            new_ids = _encode_noadd(tokenizer, stored_text)
            return new_ids, stored_text
        # prefer == "ids"
        return list(stored_ids or []), stored_text

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = int(row["group_id"])
            cid = int(row["conv_id"])
            sp = row["system_prompt"]
            stored_sys_ids = list(sp.get("ids_preview") or [])
            stored_sys_text = sp.get("text_preview")
            expected_sys = int(sp.get("expected_len") or 0)

            if prefer == "ids":
                if expected_sys > 0 and len(stored_sys_ids) != expected_sys:
                    raise ValueError(
                        f"{path}:{line_no}: system_prompt has "
                        f"{len(stored_sys_ids)} ids but expected_len="
                        f"{expected_sys}. Looks like the dump was made without "
                        f"--dump-full-content; either re-dump with that flag, "
                        f"or pass --load-from=text (if text_preview is present)."
                    )

            sys_ids, sys_text = _resolve(
                stored_sys_ids, stored_sys_text, f"{path}:{line_no} system_prompt"
            )
            sys_lens.append(len(sys_ids))
            if sys_text:
                has_any_text = True

            turns: List[TurnMaterials] = []
            for t in row.get("turns", []):
                q = t["question"]
                a = t["answer_placeholder"]
                stored_q_ids = list(q.get("ids_preview") or [])
                stored_a_ids = list(a.get("ids_preview") or [])
                stored_q_text = q.get("text_preview")
                stored_a_text = a.get("text_preview")
                q_exp = int(q.get("expected_len") or 0)
                a_exp = int(a.get("expected_len") or 0)

                if prefer == "ids":
                    if q_exp > 0 and len(stored_q_ids) != q_exp:
                        raise ValueError(
                            f"{path}:{line_no} turn {t.get('turn_idx')}: "
                            f"question ids={len(stored_q_ids)} expected={q_exp}. "
                            f"Not a full dump (or edited)."
                        )
                    if a_exp > 0 and len(stored_a_ids) != a_exp:
                        raise ValueError(
                            f"{path}:{line_no} turn {t.get('turn_idx')}: "
                            f"answer ids={len(stored_a_ids)} expected={a_exp}. "
                            f"Not a full dump (or edited)."
                        )

                q_ids, q_text = _resolve(
                    stored_q_ids, stored_q_text,
                    f"{path}:{line_no} turn {t.get('turn_idx')} question"
                )
                a_ids, a_text = _resolve(
                    stored_a_ids, stored_a_text,
                    f"{path}:{line_no} turn {t.get('turn_idx')} answer"
                )
                if q_text or a_text:
                    has_any_text = True
                q_lens.append(len(q_ids))
                a_lens.append(len(a_ids))
                turns.append(TurnMaterials(
                    turn_idx=int(t["turn_idx"]),
                    question_ids=q_ids,
                    answer_ids=a_ids,
                    question_text=q_text,
                    answer_text=a_text,
                ))
            turns.sort(key=lambda x: x.turn_idx)

            conversations.append(Conversation(
                group_id=gid,
                conv_id=cid,
                system_ids=sys_ids,
                system_text=sys_text,
                turns=turns,
                routing_key=row.get("routing_key"),
            ))
            max_group_id = max(max_group_id, gid)
            convs_per_group[gid] = max(convs_per_group.get(gid, -1), cid)
            max_turns = max(max_turns, len(turns))

    def _median(xs: List[int]) -> int:
        if not xs:
            return 0
        xs_sorted = sorted(xs)
        return xs_sorted[len(xs_sorted) // 2]

    stats = {
        "num_groups": max_group_id + 1 if max_group_id >= 0 else 0,
        "prompts_per_group": (max(convs_per_group.values()) + 1
                              if convs_per_group else 0),
        "num_turns": max_turns,
        "system_prompt_len": _median(sys_lens),
        "question_len": _median(q_lens),
        "output_len": _median(a_lens),
        "total_conversations": len(conversations),
        "total_turns": sum(len(c.turns) for c in conversations),
        "has_text": has_any_text,
    }
    return conversations, stats


# --------------------------------------------------------------------------- #
# Request senders
# --------------------------------------------------------------------------- #

@dataclass
class _TurnResult:
    ttft_ms: float
    tpot_ms: float                  # original: (last_ts - first_token_ts) / (N-1)
    e2e_ms: float
    start_ts: float
    first_token_ts: float
    end_ts: float
    prompt_len_actual: int
    output_len_actual: int
    response_ids: List[int]     # for replay-real-response mode
    response_text: str
    success: bool
    error: Optional[str]
    # Corrected TPOT: (last_ts - first_chunk_ts) / (N - first_chunk_tokens).
    # The first SSE chunk often carries multiple tokens (especially under
    # load when the server bursts post-prefill output); the original formula
    # divides by N-1 and so undercounts per-token time. This one uses the
    # actual number of tokens spanned by (last - first).
    tpot_corrected_ms: float = 0.0
    # Per-chunk arrival log for streaming responses: list of
    # (perf_counter_ts, new_tokens_in_this_chunk). Drives ITL distribution
    # and the corrected TPOT above.
    chunk_arrivals: List[Tuple[float, int]] = field(default_factory=list)


def _build_headers(cfg: Config, routing_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    else:
        env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if env_key:
            headers["Authorization"] = f"Bearer {env_key}"
    for h in cfg.extra_headers:
        k, _, v = h.partition("=")
        if k:
            headers[k.strip()] = v.strip()
    if routing_key:
        headers["X-SMG-Routing-Key"] = routing_key
    return headers


def _base_url(cfg: Config) -> str:
    host = cfg.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{cfg.port}"


def _corrected_tpot_ms(chunks: List[Tuple[float, int]]) -> float:
    """TPOT computed against the actual token span: the first chunk
    contributes its arrival time but its tokens don't count in the
    denominator, because we don't know when those tokens were generated
    individually (they arrived bundled). Matches the spirit of prom's
    per-token histogram more closely than the (last - first)/(N-1) form."""
    if len(chunks) < 2:
        return 0.0
    span_s = chunks[-1][0] - chunks[0][0]
    tokens_after_first = sum(n for _, n in chunks[1:])
    if span_s <= 0 or tokens_after_first <= 0:
        return 0.0
    return span_s * 1000.0 / tokens_after_first


def _itl_intervals_ms(chunks: List[Tuple[float, int]]) -> List[float]:
    """Per-token inter-token latency, in ms, derived from chunk arrivals.
    Each chunk after the first contributes (gap / n_tokens_in_chunk)
    repeated n_tokens_in_chunk times, so the mean across this list is
    exactly the corrected TPOT and percentiles correspond to per-token
    intervals (assuming tokens within a chunk are evenly spaced)."""
    out: List[float] = []
    for i in range(1, len(chunks)):
        prev_ts = chunks[i - 1][0]
        ts, n = chunks[i]
        if n <= 0 or ts <= prev_ts:
            continue
        per_token = (ts - prev_ts) * 1000.0 / n
        out.extend([per_token] * n)
    return out


async def _send_native_generate(session: aiohttp.ClientSession, cfg: Config,
                                prompt: Union[str, List[int]],
                                output_len: int,
                                routing_key: Optional[str]) -> _TurnResult:
    api_url = f"{_base_url(cfg)}/generate"
    field = "input_ids" if isinstance(prompt, list) else "text"
    payload: Dict[str, Any] = {
        field: prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": not cfg.disable_ignore_eos,
        },
        "stream": not cfg.disable_stream,
        "return_logprob": False,
    }
    if cfg.disable_stream:
        # For non-streaming also request ids so we can replay
        payload["return_token_ids"] = True
    if cfg.extra_request_body:
        payload.update(cfg.extra_request_body)

    headers = _build_headers(cfg, routing_key)
    start = time.perf_counter()
    ttft = 0.0
    first_token_ts = 0.0
    prompt_len_actual = 0
    output_len_actual = 0
    response_ids: List[int] = []
    response_text = ""
    last_output_len = 0
    last_ts = start
    chunk_arrivals: List[Tuple[float, int]] = []
    try:
        async with session.post(api_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                return _TurnResult(0, 0, 0, start, 0, time.perf_counter(),
                                   0, 0, [], "", False,
                                   f"HTTP {resp.status}: {body[:500]}")
            if cfg.disable_stream:
                data = await resp.json()
                end = time.perf_counter()
                meta = data.get("meta_info", {}) or {}
                prompt_len_actual = int(meta.get("prompt_tokens", 0) or 0)
                output_len_actual = int(meta.get("completion_tokens", 0) or 0)
                response_text = data.get("text", "") or ""
                # best effort: some server builds return output_ids in meta
                response_ids = list(meta.get("output_ids") or [])
                ttft_ms = (end - start) * 1000.0
                e2e_ms = ttft_ms
                tpot_ms = (e2e_ms - ttft_ms) / max(1, output_len_actual - 1) \
                    if output_len_actual > 1 else 0.0
                return _TurnResult(ttft_ms, tpot_ms, e2e_ms, start, end, end,
                                   prompt_len_actual, output_len_actual,
                                   response_ids, response_text, True, None)
            # streaming
            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                line = chunk_bytes.decode("utf-8", errors="replace")
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line in ("[DONE]", ""):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                text_now = data.get("text", "")
                meta = data.get("meta_info", {}) or {}
                ol = int(meta.get("completion_tokens", 0) or 0)
                if text_now and ttft == 0.0:
                    ttft = now - start
                    first_token_ts = now
                if ol > last_output_len:
                    chunk_arrivals.append((now, ol - last_output_len))
                    last_output_len = ol
                    last_ts = now
                if text_now:
                    response_text = text_now
                if meta.get("prompt_tokens") is not None:
                    prompt_len_actual = int(meta["prompt_tokens"])
                output_len_actual = max(output_len_actual, ol)
            end = time.perf_counter()
            e2e_ms = (end - start) * 1000.0
            ttft_ms = ttft * 1000.0 if ttft > 0 else e2e_ms
            # Original TPOT: kept for backward compatibility with prior CSVs.
            tpot_ms = ((last_ts - first_token_ts) * 1000.0
                       / max(1, output_len_actual - 1)) \
                if first_token_ts > 0 and output_len_actual > 1 else 0.0
            tpot_corr_ms = _corrected_tpot_ms(chunk_arrivals)
            return _TurnResult(ttft_ms, tpot_ms, e2e_ms, start,
                               first_token_ts, end,
                               prompt_len_actual, output_len_actual,
                               response_ids, response_text, True, None,
                               tpot_corrected_ms=tpot_corr_ms,
                               chunk_arrivals=chunk_arrivals)
    except Exception:
        end = time.perf_counter()
        return _TurnResult(0, 0, (end - start) * 1000.0, start, 0, end,
                           0, 0, [], "", False,
                           traceback.format_exc(limit=4))


async def _send_oai_chat(session: aiohttp.ClientSession, cfg: Config,
                         messages: List[Dict[str, str]],
                         output_len: int,
                         routing_key: Optional[str]) -> _TurnResult:
    api_url = f"{_base_url(cfg)}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": output_len,
        "stream": not cfg.disable_stream,
        "ignore_eos": not cfg.disable_ignore_eos,
    }
    if not cfg.disable_stream:
        payload["stream_options"] = {"include_usage": True}
    if cfg.extra_request_body:
        payload.update(cfg.extra_request_body)
    headers = _build_headers(cfg, routing_key)
    start = time.perf_counter()
    ttft = 0.0
    first_token_ts = 0.0
    prompt_len_actual = 0
    output_len_actual = 0
    response_text_parts: List[str] = []
    last_ts = start
    # OAI streaming doesn't tell us per-chunk completion_tokens, so we record
    # one entry per content-bearing chunk and reconcile against the final
    # usage count once we have it.
    chunk_ts_only: List[float] = []
    try:
        async with session.post(api_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                return _TurnResult(0, 0, 0, start, 0, time.perf_counter(),
                                   0, 0, [], "", False,
                                   f"HTTP {resp.status}: {body[:500]}")
            if cfg.disable_stream:
                data = await resp.json()
                end = time.perf_counter()
                usage = data.get("usage", {}) or {}
                prompt_len_actual = int(usage.get("prompt_tokens", 0) or 0)
                output_len_actual = int(usage.get("completion_tokens", 0) or 0)
                choices = data.get("choices") or [{}]
                msg = choices[0].get("message") or {}
                response_text = msg.get("content", "") or ""
                ttft_ms = (end - start) * 1000.0
                tpot_ms = (ttft_ms / max(1, output_len_actual - 1)) \
                    if output_len_actual > 1 else 0.0
                return _TurnResult(ttft_ms, tpot_ms, ttft_ms, start, end, end,
                                   prompt_len_actual, output_len_actual,
                                   [], response_text, True, None)
            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                line = chunk_bytes.decode("utf-8", errors="replace")
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line in ("[DONE]", ""):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                choices = data.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    c = delta.get("content")
                    if c:
                        if ttft == 0.0:
                            ttft = now - start
                            first_token_ts = now
                        response_text_parts.append(c)
                        last_ts = now
                        chunk_ts_only.append(now)
                usage = data.get("usage")
                if usage:
                    prompt_len_actual = int(usage.get("prompt_tokens") or
                                            prompt_len_actual or 0)
                    output_len_actual = int(usage.get("completion_tokens") or
                                            output_len_actual or 0)
            end = time.perf_counter()
            e2e_ms = (end - start) * 1000.0
            ttft_ms = ttft * 1000.0 if ttft > 0 else e2e_ms
            # Original TPOT formula kept as-is.
            tpot_ms = ((last_ts - first_token_ts) * 1000.0
                       / max(1, output_len_actual - 1)) \
                if first_token_ts > 0 and output_len_actual > 1 else 0.0
            # Reconcile chunk count against final usage. Best-effort
            # assumption: one token per chunk for all but the first; the
            # first chunk absorbs whatever extra tokens prefill burst out.
            chunk_arrivals: List[Tuple[float, int]] = []
            n_chunks = len(chunk_ts_only)
            if n_chunks > 0:
                if output_len_actual >= n_chunks:
                    first_n = output_len_actual - (n_chunks - 1)
                else:
                    first_n = 1
                chunk_arrivals.append((chunk_ts_only[0], first_n))
                for ts in chunk_ts_only[1:]:
                    chunk_arrivals.append((ts, 1))
            tpot_corr_ms = _corrected_tpot_ms(chunk_arrivals)
            return _TurnResult(ttft_ms, tpot_ms, e2e_ms, start,
                               first_token_ts, end,
                               prompt_len_actual, output_len_actual,
                               [], "".join(response_text_parts), True, None,
                               tpot_corrected_ms=tpot_corr_ms,
                               chunk_arrivals=chunk_arrivals)
    except Exception:
        end = time.perf_counter()
        return _TurnResult(0, 0, (end - start) * 1000.0, start, 0, end,
                           0, 0, [], "", False,
                           traceback.format_exc(limit=4))


# --------------------------------------------------------------------------- #
# Multi-turn per-conversation runner (serial turns)
# --------------------------------------------------------------------------- #

async def _run_conversation(
    session: aiohttp.ClientSession,
    cfg: Config,
    conv: Conversation,
    tokenizer,
    dump_requests_writer,
    pbar: Optional[tqdm],
) -> List[RequestRecord]:
    """Run all turns of one conversation sequentially. Returns per-turn records."""
    use_chat = cfg.backend == "sglang-oai-chat"
    records: List[RequestRecord] = []
    # Running histories. For native ids mode we keep a flat List[int]; for chat
    # we keep a messages list; for native text we keep a concatenated string.
    history_ids: List[int] = list(conv.system_ids)
    history_text_parts: List[str] = []
    if conv.system_text is not None:
        history_text_parts.append(conv.system_text)
    history_messages: List[Dict[str, str]] = []
    if use_chat:
        history_messages.append({"role": "system",
                                 "content": conv.system_text or ""})

    for t_idx, turn in enumerate(conv.turns):
        # --- build this turn's prompt by appending the current question ----
        if use_chat:
            messages = list(history_messages) + [
                {"role": "user", "content": turn.question_text or ""}
            ]
            prompt_for_send: Union[str, List[int], List[Dict[str, str]]] = messages
            expected_prompt_len = (
                conv.system_len
                + sum(c.question_len for c in conv.turns[: t_idx + 1])
                + sum(c.output_len for c in conv.turns[:t_idx])
            )  # approximate; chat template adds fixed overhead per turn
        else:
            if turn.question_text is not None:
                # native text mode
                history_text_parts.append(turn.question_text)
                prompt_for_send = "".join(history_text_parts)
                expected_prompt_len = len(_encode_noadd(tokenizer, prompt_for_send))
            else:
                # native ids mode: concatenate ids
                prompt_ids_this_turn = history_ids + list(turn.question_ids)
                prompt_for_send = prompt_ids_this_turn
                expected_prompt_len = len(prompt_ids_this_turn)

        # --- send ---------------------------------------------------------
        if use_chat:
            result = await _send_oai_chat(session, cfg, prompt_for_send,
                                          turn.output_len, conv.routing_key)
        else:
            result = await _send_native_generate(session, cfg, prompt_for_send,
                                                 turn.output_len,
                                                 conv.routing_key)

        # --- record -------------------------------------------------------
        itls = _itl_intervals_ms(result.chunk_arrivals)
        first_chunk_tokens = (
            result.chunk_arrivals[0][1] if result.chunk_arrivals else 0
        )
        rec = RequestRecord(
            group_id=conv.group_id,
            conv_id=conv.conv_id,
            turn_idx=t_idx,
            prompt_len_expected=expected_prompt_len,
            prompt_len_actual=result.prompt_len_actual,
            output_len_expected=turn.output_len,
            output_len_actual=result.output_len_actual,
            start_ts=result.start_ts,
            first_token_ts=result.first_token_ts,
            end_ts=result.end_ts,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            e2e_ms=result.e2e_ms,
            success=result.success,
            error=result.error,
            tpot_corrected_ms=result.tpot_corrected_ms,
            num_chunks=len(result.chunk_arrivals),
            first_chunk_tokens=first_chunk_tokens,
            itl_mean_ms=float(np.mean(itls)) if itls else 0.0,
            itl_p50_ms=_percentile(itls, 50),
            itl_p90_ms=_percentile(itls, 90),
            itl_p99_ms=_percentile(itls, 99),
        )
        if dump_requests_writer is not None:
            preview_ids = None
            preview_text = None
            full = cfg.dump_full_content
            if isinstance(prompt_for_send, list) and prompt_for_send and \
                    isinstance(prompt_for_send[0], int):
                preview_ids = prompt_for_send if full else prompt_for_send[:50]
            elif isinstance(prompt_for_send, str):
                preview_text = prompt_for_send if full else prompt_for_send[:200]
            elif isinstance(prompt_for_send, list):
                # chat messages
                blob = json.dumps(prompt_for_send, ensure_ascii=False)
                preview_text = blob if full else blob[:400]
            rec.prompt_ids_preview = preview_ids
            rec.prompt_text_preview = preview_text

        records.append(rec)
        if dump_requests_writer is not None:
            dump_requests_writer.write(
                json.dumps(rec.to_dict(), ensure_ascii=False) + "\n"
            )
        if pbar is not None:
            pbar.update(1)

        # --- advance history for next turn --------------------------------
        if use_chat:
            if cfg.replay_real_response and result.success and result.response_text:
                assistant_text = result.response_text
            else:
                assistant_text = turn.answer_text or ""
            history_messages.append(
                {"role": "user", "content": turn.question_text or ""}
            )
            history_messages.append(
                {"role": "assistant", "content": assistant_text}
            )
        else:
            if turn.question_text is not None:
                # native text mode
                if cfg.replay_real_response and result.success and result.response_text:
                    history_text_parts.append(result.response_text)
                else:
                    history_text_parts.append(turn.answer_text or "")
            else:
                # native ids mode
                history_ids.extend(turn.question_ids)
                if cfg.replay_real_response and result.success and result.response_ids:
                    # trim/pad to expected output_len to keep length deterministic
                    resp = result.response_ids[: turn.output_len]
                    history_ids.extend(resp)
                else:
                    history_ids.extend(turn.answer_ids)

    return records


# --------------------------------------------------------------------------- #
# Scheduler: concurrency + rate + shuffle
# --------------------------------------------------------------------------- #

async def _conversation_arrival(conversations: List[Conversation],
                                request_rate: float,
                                num_turns: int):
    """Yield conversations at arrival times.

    Each conversation enters the system at a Poisson-spaced arrival time based
    on request_rate. Since each conversation is `num_turns` requests, we scale
    the interval so that total request rate ~= request_rate.
    """
    if request_rate == float("inf") or request_rate <= 0 or num_turns <= 0:
        for conv in conversations:
            yield conv
        return
    conv_rate = request_rate / max(1, num_turns)
    for conv in conversations:
        yield conv
        interval = np.random.exponential(1.0 / conv_rate)
        await asyncio.sleep(float(interval))


async def run_benchmark(cfg: Config, tokenizer,
                        conversations: List[Conversation]) -> Tuple[
                            List[RequestRecord], float]:
    # ordering
    if not cfg.ordered:
        rng = random.Random(cfg.seed + 101)
        rng.shuffle(conversations)

    # cap total requests via trimming turns
    total_target = cfg.num_prompts
    if total_target < cfg.total_requests:
        # Trim conversations until total turn count <= target
        trimmed: List[Conversation] = []
        remaining = total_target
        for conv in conversations:
            if remaining <= 0:
                break
            if remaining >= len(conv.turns):
                trimmed.append(conv)
                remaining -= len(conv.turns)
            else:
                conv.turns = conv.turns[:remaining]
                trimmed.append(conv)
                remaining = 0
        conversations = trimmed

    # dump requests writer (if enabled)
    dump_requests_fp = None
    if cfg.dump_prompts_dir:
        Path(cfg.dump_prompts_dir).mkdir(parents=True, exist_ok=True)
        dump_requests_fp = open(
            Path(cfg.dump_prompts_dir) / "requests.jsonl",
            "w", encoding="utf-8",
        )

    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT_SEC)
    all_records: List[RequestRecord] = []

    total_turns = sum(len(c.turns) for c in conversations)
    pbar = tqdm(total=total_turns, desc="turns")

    t_start = time.perf_counter()

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        read_bufsize=AIOHTTP_READ_BUFSIZE,
    ) as session:

        async def _one_conv(conv: Conversation):
            async with semaphore:
                recs = await _run_conversation(
                    session, cfg, conv, tokenizer, dump_requests_fp, pbar,
                )
                all_records.extend(recs)

        tasks: List[asyncio.Task] = []
        async for conv in _conversation_arrival(
            conversations, cfg.request_rate, cfg.num_turns
        ):
            tasks.append(asyncio.create_task(_one_conv(conv)))
        if tasks:
            await asyncio.gather(*tasks)

    pbar.close()
    duration_s = time.perf_counter() - t_start

    if dump_requests_fp is not None:
        dump_requests_fp.close()

    return all_records, duration_s


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


@dataclass
class Summary:
    case_name: str
    case_start_time: str
    num_groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    num_turns: int
    num_prompts: int
    request_rate: float
    max_concurrency: int
    completed: int
    failed: int
    duration_s: float
    # Active span = max(end_ts) - min(start_ts) across successful requests.
    # Excludes idle padding before the first request and after the last one,
    # so it lines up with what Prometheus shows during the actual load window.
    active_duration_s: float
    request_throughput: float
    input_throughput: float
    output_throughput: float
    request_throughput_active: float
    input_throughput_active: float
    output_throughput_active: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    # Corrected TPOT — denominator excludes the bundled first-chunk tokens.
    mean_tpot_corrected_ms: float
    median_tpot_corrected_ms: float
    p90_tpot_corrected_ms: float
    p99_tpot_corrected_ms: float
    # Inter-token latency aggregated across all chunk gaps from all
    # successful requests; closest analogue to prom's per-token histogram.
    mean_itl_ms: float
    median_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    # Streaming diagnostics — helps explain TPOT discrepancies.
    mean_first_chunk_tokens: float
    mean_tokens_per_chunk: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    mean_prompt_len_expected: float
    mean_prompt_len_actual: float
    mean_prompt_len_diff: float


def _compute_summary(cfg: Config, records: List[RequestRecord],
                     duration_s: float) -> Summary:
    ok = [r for r in records if r.success]
    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    tpot_corrs = [r.tpot_corrected_ms for r in ok if r.tpot_corrected_ms > 0]
    e2es = [r.e2e_ms for r in ok]
    prompt_exp = [r.prompt_len_expected for r in ok]
    prompt_act = [r.prompt_len_actual for r in ok if r.prompt_len_actual > 0]
    output_act_sum = sum(r.output_len_actual for r in ok)
    input_act_sum = sum(r.prompt_len_actual for r in ok)
    # Aggregate ITL across requests by averaging per-request percentiles
    # (per-token raw intervals would need re-collection; per-request
    # percentiles are a fine proxy and cheap to compute).
    itl_means = [r.itl_mean_ms for r in ok if r.itl_mean_ms > 0]
    itl_p50s = [r.itl_p50_ms for r in ok if r.itl_p50_ms > 0]
    itl_p90s = [r.itl_p90_ms for r in ok if r.itl_p90_ms > 0]
    itl_p99s = [r.itl_p99_ms for r in ok if r.itl_p99_ms > 0]
    first_chunk_toks = [r.first_chunk_tokens for r in ok if r.num_chunks > 0]
    tokens_per_chunk = [
        r.output_len_actual / r.num_chunks
        for r in ok
        if r.num_chunks > 0 and r.output_len_actual > 0
    ]

    duration = max(duration_s, 1e-9)
    req_tp = len(ok) / duration
    in_tp = input_act_sum / duration
    out_tp = output_act_sum / duration

    # Active span: from the first successful request's send time to the last
    # successful request's completion. Throughputs computed against this span
    # exclude the idle warm-up/teardown lag and line up with Prometheus
    # (which only sees the server-side window where work was actually
    # happening).
    if ok:
        active_dur = max(max(r.end_ts for r in ok) - min(r.start_ts for r in ok),
                         1e-9)
    else:
        active_dur = 0.0
    active_dur_safe = max(active_dur, 1e-9)
    req_tp_act = len(ok) / active_dur_safe
    in_tp_act = input_act_sum / active_dur_safe
    out_tp_act = output_act_sum / active_dur_safe

    mean_exp = float(np.mean(prompt_exp)) if prompt_exp else 0.0
    mean_act = float(np.mean(prompt_act)) if prompt_act else 0.0

    return Summary(
        case_name=cfg.case_name or "unnamed",
        case_start_time=datetime.now().isoformat(timespec="seconds"),
        num_groups=cfg.num_groups,
        prompts_per_group=cfg.prompts_per_group,
        system_prompt_len=cfg.system_prompt_len,
        question_len=cfg.question_len,
        output_len=cfg.output_len,
        num_turns=cfg.num_turns,
        num_prompts=cfg.num_prompts,
        request_rate=cfg.request_rate if cfg.request_rate != float("inf") else -1.0,
        max_concurrency=cfg.max_concurrency,
        completed=len(ok),
        failed=len(records) - len(ok),
        duration_s=duration_s,
        active_duration_s=active_dur,
        request_throughput=req_tp,
        input_throughput=in_tp,
        output_throughput=out_tp,
        request_throughput_active=req_tp_act,
        input_throughput_active=in_tp_act,
        output_throughput_active=out_tp_act,
        mean_ttft_ms=float(np.mean(ttfts)) if ttfts else 0.0,
        median_ttft_ms=_percentile(ttfts, 50),
        p90_ttft_ms=_percentile(ttfts, 90),
        p99_ttft_ms=_percentile(ttfts, 99),
        mean_tpot_ms=float(np.mean(tpots)) if tpots else 0.0,
        median_tpot_ms=_percentile(tpots, 50),
        p90_tpot_ms=_percentile(tpots, 90),
        p99_tpot_ms=_percentile(tpots, 99),
        mean_tpot_corrected_ms=float(np.mean(tpot_corrs)) if tpot_corrs else 0.0,
        median_tpot_corrected_ms=_percentile(tpot_corrs, 50),
        p90_tpot_corrected_ms=_percentile(tpot_corrs, 90),
        p99_tpot_corrected_ms=_percentile(tpot_corrs, 99),
        mean_itl_ms=float(np.mean(itl_means)) if itl_means else 0.0,
        median_itl_ms=float(np.mean(itl_p50s)) if itl_p50s else 0.0,
        p90_itl_ms=float(np.mean(itl_p90s)) if itl_p90s else 0.0,
        p99_itl_ms=float(np.mean(itl_p99s)) if itl_p99s else 0.0,
        mean_first_chunk_tokens=float(np.mean(first_chunk_toks)) if first_chunk_toks else 0.0,
        mean_tokens_per_chunk=float(np.mean(tokens_per_chunk)) if tokens_per_chunk else 0.0,
        mean_e2e_latency_ms=float(np.mean(e2es)) if e2es else 0.0,
        median_e2e_latency_ms=_percentile(e2es, 50),
        p90_e2e_latency_ms=_percentile(e2es, 90),
        p99_e2e_latency_ms=_percentile(e2es, 99),
        mean_prompt_len_expected=mean_exp,
        mean_prompt_len_actual=mean_act,
        mean_prompt_len_diff=mean_act - mean_exp,
    )


def _write_per_request_jsonl(path: str, records: List[RequestRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _append_summary_csv(path: str, summary: Summary) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    row = asdict(summary)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_content_dump(path: str, conversations: List[Conversation],
                        tokenizer, full: bool = False) -> None:
    """Dump the generated materials for every (group, conv).

    When `full` is False (default), each ids/text field is clipped to 50
    ids / 200 chars. When True, the full content is dumped (file size grows
    linearly with total tokens) AND ids are decoded back to text when the
    original text was not available (native ids path), so the file is
    human-readable and editable.
    """
    def _clip_ids(ids):
        return ids if full else ids[:50]

    def _clip_text(txt):
        if txt is None:
            return None
        return txt if full else txt[:200]

    def _ensure_text(ids, text):
        """When dumping full content, make sure a text form exists so the file
        is editable. If text is None/empty and ids are present, decode on the
        fly (lossy: BPE decode of random/tile ids may not be a perfect inverse,
        but it is human-readable, which is the point)."""
        if not full:
            return _clip_text(text)
        if text:
            return text
        if ids:
            try:
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                return None
        return None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            sys_actual_len = len(conv.system_ids) if conv.system_ids else (
                len(_encode_noadd(tokenizer, conv.system_text or "")))
            row = {
                "group_id": conv.group_id,
                "conv_id": conv.conv_id,
                "routing_key": conv.routing_key,
                "system_prompt": {
                    "expected_len": len(conv.system_ids),
                    "actual_len": sys_actual_len,
                    "ids_preview": _clip_ids(conv.system_ids),
                    "text_preview": _ensure_text(conv.system_ids,
                                                 conv.system_text),
                },
                "turns": [],
            }
            for t in conv.turns:
                q_actual = len(t.question_ids) if t.question_ids else (
                    len(_encode_noadd(tokenizer, t.question_text or "")))
                a_actual = len(t.answer_ids) if t.answer_ids else (
                    len(_encode_noadd(tokenizer, t.answer_text or "")))
                row["turns"].append({
                    "turn_idx": t.turn_idx,
                    "question": {
                        "expected_len": len(t.question_ids),
                        "actual_len": q_actual,
                        "ids_preview": _clip_ids(t.question_ids),
                        "text_preview": _ensure_text(t.question_ids,
                                                     t.question_text),
                    },
                    "answer_placeholder": {
                        "expected_len": len(t.answer_ids),
                        "actual_len": a_actual,
                        "ids_preview": _clip_ids(t.answer_ids),
                        "text_preview": _ensure_text(t.answer_ids,
                                                     t.answer_text),
                    },
                })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(summary: Summary) -> None:
    print()
    print("=" * 60)
    print(f"{'Multi-turn benchmark summary':^60}")
    print("=" * 60)
    print(f"{'Case':<30}{summary.case_name}")
    print(f"{'Completed / total':<30}"
          f"{summary.completed} / {summary.completed + summary.failed}")
    print(f"{'Duration wall (s)':<30}{summary.duration_s:.2f}")
    print(f"{'Duration active (s)':<30}{summary.active_duration_s:.2f}")
    print(f"{'Req throughput wall (r/s)':<30}{summary.request_throughput:.3f}")
    print(f"{'Req throughput active (r/s)':<30}"
          f"{summary.request_throughput_active:.3f}")
    print(f"{'In throughput wall (tok/s)':<30}{summary.input_throughput:.1f}")
    print(f"{'In throughput active (tok/s)':<30}"
          f"{summary.input_throughput_active:.1f}")
    print(f"{'Out throughput wall (tok/s)':<30}{summary.output_throughput:.1f}")
    print(f"{'Out throughput active (tok/s)':<30}"
          f"{summary.output_throughput_active:.1f}")
    print("-" * 60)
    print(f"{'Mean TTFT (ms)':<30}{summary.mean_ttft_ms:.2f}")
    print(f"{'Median TTFT (ms)':<30}{summary.median_ttft_ms:.2f}")
    print(f"{'P90 TTFT (ms)':<30}{summary.p90_ttft_ms:.2f}")
    print(f"{'P99 TTFT (ms)':<30}{summary.p99_ttft_ms:.2f}")
    print(f"{'Mean TPOT (ms)':<30}{summary.mean_tpot_ms:.2f}")
    print(f"{'Median TPOT (ms)':<30}{summary.median_tpot_ms:.2f}")
    print(f"{'P90 TPOT (ms)':<30}{summary.p90_tpot_ms:.2f}")
    print(f"{'P99 TPOT (ms)':<30}{summary.p99_tpot_ms:.2f}")
    print(f"{'Mean TPOT* (ms)':<30}{summary.mean_tpot_corrected_ms:.2f}")
    print(f"{'Median TPOT* (ms)':<30}{summary.median_tpot_corrected_ms:.2f}")
    print(f"{'P90 TPOT* (ms)':<30}{summary.p90_tpot_corrected_ms:.2f}")
    print(f"{'P99 TPOT* (ms)':<30}{summary.p99_tpot_corrected_ms:.2f}")
    print(f"{'Mean ITL (ms)':<30}{summary.mean_itl_ms:.2f}")
    print(f"{'Median ITL (ms)':<30}{summary.median_itl_ms:.2f}")
    print(f"{'P90 ITL (ms)':<30}{summary.p90_itl_ms:.2f}")
    print(f"{'P99 ITL (ms)':<30}{summary.p99_itl_ms:.2f}")
    print(f"{'Mean first-chunk tokens':<30}"
          f"{summary.mean_first_chunk_tokens:.2f}")
    print(f"{'Mean tokens/chunk':<30}{summary.mean_tokens_per_chunk:.2f}")
    print("  (TPOT* uses (last - first) / (N - first_chunk_tokens); ITL is "
          "per-token spread.)")
    print(f"{'Mean E2E (ms)':<30}{summary.mean_e2e_latency_ms:.2f}")
    print(f"{'Median E2E (ms)':<30}{summary.median_e2e_latency_ms:.2f}")
    print(f"{'P90 E2E (ms)':<30}{summary.p90_e2e_latency_ms:.2f}")
    print(f"{'P99 E2E (ms)':<30}{summary.p99_e2e_latency_ms:.2f}")
    print("-" * 60)
    print(f"{'Mean prompt len (expected)':<30}"
          f"{summary.mean_prompt_len_expected:.1f}")
    print(f"{'Mean prompt len (actual)':<30}"
          f"{summary.mean_prompt_len_actual:.1f}")
    print(f"{'Mean diff':<30}{summary.mean_prompt_len_diff:+.1f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    # seed global RNGs so any residual randomness is also controlled
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"[bench_multi_turn] loading tokenizer: {cfg.tokenizer}")
    tokenizer = get_tokenizer(cfg.tokenizer)

    if cfg.load_dataset:
        print(f"[bench_multi_turn] loading pre-generated dataset: "
              f"{cfg.load_dataset} (load_from={cfg.load_from})")
        conversations, stats = load_conversations_from_dump(
            cfg.load_dataset, tokenizer=tokenizer, prefer=cfg.load_from,
        )
        # Overwrite cfg with the dataset's actual shape (for summary/reporting).
        cfg.num_groups = stats["num_groups"]
        cfg.prompts_per_group = stats["prompts_per_group"]
        cfg.num_turns = stats["num_turns"]
        cfg.system_prompt_len = stats["system_prompt_len"]
        cfg.question_len = stats["question_len"]
        cfg.output_len = stats["output_len"]
        total = stats["total_turns"]
        if cfg.num_prompts <= 0 or cfg.num_prompts > total:
            cfg.num_prompts = total
        print(f"[bench_multi_turn] loaded {stats['total_conversations']} "
              f"conversations / {stats['total_turns']} turns "
              f"(groups={cfg.num_groups}, prompts_per_group={cfg.prompts_per_group}, "
              f"num_turns={cfg.num_turns}, "
              f"median sys/Q/A lens = "
              f"{cfg.system_prompt_len}/{cfg.question_len}/{cfg.output_len})")
        # compatibility check for chat backend
        if cfg.backend == "sglang-oai-chat" and not stats["has_text"]:
            raise ValueError(
                "--backend=sglang-oai-chat requires text content, but the "
                "loaded dataset has none (it was dumped from a native ids "
                "run). Re-dump from a synthetic-text or a chat-backend run, "
                "or switch to --backend sglang."
            )
    else:
        print(f"[bench_multi_turn] generating conversations "
              f"(mode={cfg.dataset_mode}, groups={cfg.num_groups}, "
              f"prompts_per_group={cfg.prompts_per_group}, "
              f"turns={cfg.num_turns})")
        generator = build_generator(cfg, tokenizer)
        conversations = generate_conversations(cfg, tokenizer, generator)

    # dump content if requested
    if cfg.dump_prompts_dir:
        Path(cfg.dump_prompts_dir).mkdir(parents=True, exist_ok=True)
        content_path = str(Path(cfg.dump_prompts_dir) / "content.jsonl")
        print(f"[bench_multi_turn] dumping content to {content_path}")
        _write_content_dump(content_path, conversations, tokenizer,
                            full=cfg.dump_full_content)

    print(f"[bench_multi_turn] launching benchmark "
          f"(backend={cfg.backend}, target={_base_url(cfg)}, "
          f"concurrency={cfg.max_concurrency}, rate={cfg.request_rate})")
    records, duration_s = asyncio.run(
        run_benchmark(cfg, tokenizer, conversations)
    )

    summary = _compute_summary(cfg, records, duration_s)
    print_summary(summary)

    if cfg.output_file:
        _write_per_request_jsonl(cfg.output_file, records)
        print(f"[bench_multi_turn] per-request jsonl -> {cfg.output_file}")
    if cfg.summary_csv:
        _append_summary_csv(cfg.summary_csv, summary)
        print(f"[bench_multi_turn] summary row appended -> {cfg.summary_csv}")

    # non-zero exit if any request failed
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

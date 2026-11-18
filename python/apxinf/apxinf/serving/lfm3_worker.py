"""Run one resident EngineTailor LFM3 model through the worker/2.0 protocol.

This is a fork of ``python/apxinf/apxinf/serving/text_worker.py`` (base branch
``codex/serving-split-3-text-provider``): the protocol shell, validation and
settlement logic are unchanged, and ``MLXRuntime`` is replaced by an
``LFM3Runtime`` that drives EngineTailor's LFM3 MLX ``Engine`` (packed4
official checkpoint) instead of mlx-lm.

Cross-turn prefix reuse (the AG1 addition, internal to the worker so the
worker/2.0 wire profile is unchanged): every submitted prompt is compared with
the token sequence whose KV/conv state is currently resident. The longest
common prefix is reused exactly:

* full-attention layers: the per-layer KV arrays are truncated at the boundary;
* sliding-attention layers: the resident cache keeps the full history (engine
  flag ``_keep_full_sliding``), so the last ``window-1`` keys before the
  boundary can be reconstructed by truncation;
* convolution layers: a snapshot of the 13 layers' conv state is stored every
  ``--snapshot-interval`` tokens; restoring at the newest snapshot <= boundary
  replays at most that many tokens.

With ``--prefix-cache off`` the runtime resets before every prompt, which is
the AG3 control arm. Every submitted request appends one JSON line to
``--ledger`` (the data source AG2/AG3 use for f and the time split).

The main thread owns all MLX objects. Reader and writer threads handle bytes only.
This file also supports direct execution without importing the policy package.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import resource
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable

import jinja2
import numpy as np
from tokenizers import Tokenizer

if __package__:
    from . import contracts
else:
    import contracts


ADAPTER_REVISION = "ag1-lfm3-prefix-v1"
PINNED_RUNTIME = {"python": "3.11.14", "mlx": "0.31.2", "tokenizers": "0.22.2"}
MAX_FRAME_BYTES = 1_048_576
MAX_COMMANDS = 10_000
BOS_TOKEN = "<|startoftext|>"
ENGINE_COMMIT = "d181fbb1"
GENERATION_TAGS = ("{%- generation -%}", "{%- endgeneration -%}")


class WorkerFailure(Exception):
    """Report a stable error without dependency details or input text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GenerationStopped(Exception):
    """Return control from generation at a safe boundary."""


class ChannelFailure(Exception):
    """Stop a worker whose protocol channel cannot remain valid."""


@dataclass
class InvalidCommand:
    """Keep a valid command envelope with invalid request content."""

    frame: dict[str, Any]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_manifest(model_path: Path, runtime: dict[str, str], execution: dict[str, Any]) -> dict[str, Any]:
    """Bind local model artifacts, the interpreter, and this adapter by content."""
    artifacts = []
    entries = sorted(model_path.iterdir(), key=lambda path: path.name)
    if not entries or len(entries) > 4096:
        raise WorkerFailure("model_unavailable", "The model file count is invalid.")
    for path in entries:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise WorkerFailure("model_unavailable", "Each model artifact must resolve to a regular file.")
        digest = _file_digest(resolved)
        after = resolved.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns
        ) or path.resolve(strict=True) != resolved:
            raise WorkerFailure("model_unavailable", "A model file changed during identity checks.")
        artifacts.append({"path": path.name, "size_bytes": before.st_size, "sha256": digest})
    return {
        "artifacts": artifacts,
        "runtime": runtime,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_sha256": _file_digest(Path(__file__).resolve()),
        "python_sha256": _file_digest(Path(sys.executable).resolve()),
        "execution": execution,
    }


class LFM3Runtime:
    """EngineTailor LFM3 ``Engine`` on the owner thread, with a prefix cache."""

    def __init__(self, model_path: str, max_context: int, prefill_step_size: int,
                 output_batch_tokens: int = 1, memory_limit_bytes: int = 10 * 1024**3,
                 prefix_cache: bool = True, ledger_path: str | None = None,
                 engine_root: str | None = None, snapshot_interval: int = 256):
        self.owner = threading.get_ident()
        self.model_path = str(Path(model_path).resolve(strict=True))
        self.prefix_cache = bool(prefix_cache)
        self.prefill_step = max(1, int(prefill_step_size))
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.ledger_path = Path(ledger_path).resolve() if ledger_path else None
        self.runtime = {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": f"enginetailor-{ENGINE_COMMIT}",
        }
        if (self.runtime["python"], self.runtime["mlx"]) != (
            PINNED_RUNTIME["python"], PINNED_RUNTIME["mlx"]
        ):
            raise WorkerFailure("model_unavailable", "The runtime does not match the pinned versions.")
        if importlib.metadata.version("tokenizers") != PINNED_RUNTIME["tokenizers"]:
            raise WorkerFailure("model_unavailable", "The tokenizer package does not match the pinned version.")
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        root = (Path(engine_root).resolve() if engine_root
                else Path(__file__).resolve().parents[4] / "engine")
        if not (root / "enginetailor" / "models" / "lfm3" / "runtime.py").is_file():
            raise WorkerFailure("model_unavailable", "The EngineTailor LFM3 tree is unavailable.")
        sys.path.insert(0, str(root))

        self.manifest = model_manifest(Path(self.model_path), self.runtime, {
            "provider": "enginetailor-lfm3", "precision": "packed4",
            "prefill_step_size": prefill_step_size,
            "output_batch_tokens": output_batch_tokens,
            "memory_limit_bytes": memory_limit_bytes,
            "engine": "EngineTailor", "engine_commit": ENGINE_COMMIT,
            "prefix_cache": bool(prefix_cache),
            "snapshot_interval": self.snapshot_interval,
        })
        self.manifest["tokenizers"] = importlib.metadata.version("tokenizers")
        self.manifest["jinja2"] = importlib.metadata.version("jinja2")
        self.manifest["contracts_sha256"] = _file_digest(Path(contracts.__file__).resolve())
        self.manifest["engine_root"] = str(root)

        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            from enginetailor.models.lfm3.runtime import Engine

            self.mx = mx
            mx.set_memory_limit(memory_limit_bytes)
            mx.reset_peak_memory()
            self.engine = Engine(root=self.model_path, dtype="bfloat16", quant="packed4")
            # Patched-engine switch: retain the full sliding-attention history so
            # a boundary inside an earlier turn can still be reconstructed.
            self.engine._keep_full_sliding = True
            if not getattr(self.engine, "_keep_full_sliding", False):
                raise WorkerFailure("model_unavailable", "The engine lacks the prefix-cache patch.")
            tok_path = Path(self.model_path) / "tokenizer.json"
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.chat_template = (Path(self.model_path) / "chat_template.jinja").read_text()
            for tag in GENERATION_TAGS:
                self.chat_template = self.chat_template.replace(tag, "")
            env = jinja2.Environment()
            env.globals["raise_exception"] = _raise_template_error
            self._template = env.from_string(self.chat_template)

        self.max_context = min(int(max_context), int(self.engine.cfg.max_position_embeddings))
        self.vocab_size = int(self.engine.cfg.vocab_size)
        model_config = json.loads((Path(self.model_path) / "config.json").read_text())
        eos = model_config.get("eos_token_id")
        if isinstance(eos, int):
            eos = [eos]
        if not eos:
            raise WorkerFailure("model_unavailable", "The checkpoint declares no EOS token.")
        self.eos_token_ids = sorted({int(token) for token in eos})

        # resident prefix state
        self.cache_tokens: list[int] = []
        self._snapshots: list[tuple[int, tuple[Any, ...]]] = []
        self._session_id: str | None = None
        self._last_request_id: str | None = None
        self._out_ids: list[int] = []
        self._emitted = ""
        self._stats: dict[str, Any] = {}

    # -- protocol runtime surface --------------------------------------- #

    def _check_owner(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the execution owner can use MLX objects.")

    def prepare(self, command: dict[str, Any]) -> list[int]:
        self._check_owner()
        with contextlib.redirect_stdout(sys.stderr):
            text = self._template.render(
                bos_token=BOS_TOKEN,
                messages=command["messages"],
                tools=command["tools"] or [],
                add_generation_prompt=True,
                **command["template_options"],
            )
            tokens = self.tokenizer.encode(text).ids
        return [int(token) for token in tokens]

    def generate(self, command: dict[str, Any], progress: Callable[[int], None]):
        self._check_owner()
        prompt = [int(token) for token in command["token_ids"]]
        max_tokens = int(command["max_tokens"])
        self._out_ids = []
        self._emitted = ""
        self.mx.reset_peak_memory()
        stats: dict[str, Any] = {
            "prompt_tokens": len(prompt), "cache_hit_tokens": 0, "cache_hit": False,
            "miss_reason": None, "prefill_tokens": 0, "replay_tokens": 0,
            "prefill_ms": 0.0, "decode_ms": 0.0, "generated_tokens": 0,
            "start_position": 0, "session_id": None, "parent_request_id": None,
            "cache_tokens_after": 0, "snapshots": 0,
        }
        self._stats = stats
        self._last_request_id = command["request_id"]
        start = self._admit_prompt(prompt, stats)

        prefill_t0 = time.perf_counter()
        logits = None
        position = start
        while position < len(prompt):
            end = min(position + self.prefill_step, len(prompt))
            logits = self.engine.prefill(
                prompt[position:end], last_only=end == len(prompt),
                positions=list(range(position, end)),
            )
            position = end
            self.cache_tokens = prompt[:position]
            self._snapshot_if_due()
            progress(position)
        stats["prefill_ms"] = (time.perf_counter() - prefill_t0) * 1e3
        stats["prefill_tokens"] = len(prompt) - start
        stats["replay_tokens"] = max(0, min(stats["cache_hit_tokens"], len(prompt)) - start)
        if logits is None:
            raise WorkerFailure("internal_error", "Prefill produced no logits.")

        token = _argmax(logits)
        while True:
            yield token
            stats["generated_tokens"] += 1
            if stats["generated_tokens"] >= max_tokens or token in self.eos_token_ids:
                break
            step_t0 = time.perf_counter()
            logits = self.engine.step(token)
            stats["decode_ms"] += (time.perf_counter() - step_t0) * 1e3
            # The emitted token is now in the KV/conv state; the final emitted
            # token stays out of the cache (the next prompt re-prefills it).
            self.cache_tokens.append(token)
            self._snapshot_if_due()
            token = _argmax(logits)
        stats["cache_tokens_after"] = len(self.cache_tokens)
        stats["snapshots"] = len(self._snapshots)

    def decode(self, token: int, eos: bool = False) -> str:
        self._check_owner()
        self._out_ids.append(int(token))
        if eos:
            return ""
        return self._text_delta(final=False)

    def finish_text(self) -> str:
        self._check_owner()
        return self._text_delta(final=True)

    def _text_delta(self, *, final: bool) -> str:
        full = self.tokenizer.decode(self._out_ids, skip_special_tokens=True)
        if not final and full.endswith("\ufffd"):
            full = full[:-1]          # hold back a partial UTF-8 sequence
        delta = full[len(self._emitted):]
        self._emitted = full
        return delta

    @property
    def consumed_position(self) -> int:
        return len(self.cache_tokens)

    def memory(self) -> dict[str, int]:
        self._check_owner()
        return {
            "active_bytes": int(self.mx.get_active_memory()),
            "peak_bytes": int(self.mx.get_peak_memory()),
            "cache_bytes": int(self.mx.get_cache_memory()),
        }

    def inspect_memory(self) -> dict[str, Any]:
        self._check_owner()
        return {
            "allocator": self.memory(),
            "runtime_consumed_position": len(self.cache_tokens),
            "cache_payload_bytes": None,
            "layers": [],
        }

    def settle(self) -> None:
        """Keep the prefix state resident; release only transient buffers."""
        self._check_owner()
        self.mx.clear_cache()

    # -- prefix-cache internals ----------------------------------------- #

    def _admit_prompt(self, prompt: list[int], stats: dict[str, Any]) -> int:
        """Choose the start position and restore the three cache states."""
        cached = self.cache_tokens
        hit = 0
        session = None
        parent = None
        if not self.prefix_cache:
            self._reset_state()
            reason, start = "prefix_cache_off", 0
        elif not cached:
            reason, start = "first_turn", 0
            session = uuid.uuid4().hex
        else:
            hit = _common_prefix(cached, prompt)
            if hit >= len(cached) and hit < len(prompt):
                reason, start = None, len(cached)
                session, parent = self._session_id, self._last_request_id
            elif hit >= len(prompt):
                hit = len(prompt) - 1
                snapshot = self._snapshot_at_or_before(hit)
                if snapshot is None:
                    self._reset_state()
                    reason, start, hit = "cached_prompt_reset", 0, 0
                    session = uuid.uuid4().hex
                else:
                    self._restore(snapshot)
                    reason, start = "cached_prompt", snapshot
                    session, parent = self._session_id, self._last_request_id
            elif hit == 0:
                self._reset_state()
                reason, start = "prefix_rewrite", 0
                session = uuid.uuid4().hex
            else:
                snapshot = self._snapshot_at_or_before(hit)
                if snapshot is None:
                    self._reset_state()
                    reason, start, hit = "snapshot_gap", 0, 0
                    session = uuid.uuid4().hex
                else:
                    self._restore(snapshot)
                    reason, start = "prefix_rewrite", snapshot
                    session, parent = self._session_id, self._last_request_id
        if session is not None:
            self._session_id = session
        stats.update({
            "cache_hit_tokens": hit, "cache_hit": hit > 0, "miss_reason": reason,
            "start_position": start, "session_id": self._session_id,
            "parent_request_id": parent,
            # A restored boundary replays (hit - start) cached tokens; a pure
            # append replays nothing and start == hit.
            "replay_tokens": max(0, hit - start),
        })
        return start

    def _reset_state(self) -> None:
        self.engine.reset()
        self.cache_tokens = []
        self._snapshots = []
        self._session_id = None

    def _snapshot_if_due(self) -> None:
        position = int(self.engine._pos)
        if position <= 0:
            return
        if self._snapshots and position - self._snapshots[-1][0] < self.snapshot_interval:
            return
        conv = tuple(self.engine._conv_state[i] for i in range(self.engine.cfg.num_hidden_layers))
        self._snapshots.append((position, conv))

    def _snapshot_at_or_before(self, position: int) -> int | None:
        best = None
        for snapshot_position, _ in self._snapshots:
            if snapshot_position <= position:
                best = snapshot_position
            else:
                break
        return best

    def _restore(self, position: int) -> None:
        """Restore KV, sliding window and conv state at ``position``."""
        engine = self.engine
        for index, entry in enumerate(engine._kv):
            if entry is None:
                continue
            # The prefix cache relies on every resident layer spanning [0, _pos);
            # a short cache means the engine dropped history and cannot be rewound.
            if entry[0].shape[2] != engine._pos:
                raise RuntimeError("resident cache length does not match its position")
            # Keep the whole prefix, not only the sliding window: the window is
            # applied by the attention mask, and a later rewind may need keys
            # older than the window at this boundary.
            engine._kv[index] = (entry[0][:, :, :position, :],
                                 entry[1][:, :, :position, :])
        snapshot = dict((pos, conv) for pos, conv in self._snapshots)
        conv_states = snapshot.get(position)
        if conv_states is None:
            raise RuntimeError("prefix snapshot is missing")
        for index, state in enumerate(conv_states):
            engine._conv_state[index] = state
        engine._pos = position
        engine._chunk_start = 0
        self.cache_tokens = self.cache_tokens[:position]
        self._snapshots = [(pos, conv) for pos, conv in self._snapshots if pos <= position]

    # -- ledger ---------------------------------------------------------- #

    def write_ledger(self, attempt: Any, status: str, cause: str,
                     usage: dict[str, Any], metrics: dict[str, Any],
                     executed: bool = True) -> None:
        if self.ledger_path is None:
            return
        stats = (self._stats or {}) if executed else {}
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        line = {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "request_id": attempt.key[0],
            "attempt": attempt.key[1],
            "session_id": stats.get("session_id") or "none",
            "parent_request_id": stats.get("parent_request_id"),
            "status": status,
            "cause": cause,
            "prompt_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
            "cache_hit": bool(stats.get("cache_hit")),
            "cache_hit_tokens": int(stats.get("cache_hit_tokens", 0)),
            "prefill_tokens": int(stats.get("prefill_tokens", 0)),
            "replay_tokens": int(stats.get("replay_tokens", 0)),
            "prefill_ms": round(float(stats.get("prefill_ms", 0.0)), 3),
            "decode_ms": round(float(stats.get("decode_ms", 0.0)), 3),
            "first_token_ms": round(float(metrics.get("ttft_ns", 0)) / 1e6, 3),
            "miss_reason": stats.get("miss_reason") or "not_executed",
            "ane_used": bool(self.engine.ane_enabled),
            "ane_fraction": (self.engine.ane_stats() or {}).get("fraction") if self.engine.ane_enabled else None,
            "quant": self.engine.quant_mode,
            "peak_memory_bytes": int(metrics.get("peak_memory_bytes", 0)),
            "rss_peak_bytes": peak_rss,
            "swap_used_bytes": _swap_used_bytes(),
            "cache_tokens_after": int(stats.get("cache_tokens_after", len(self.cache_tokens))),
            "engine_position": int(self.engine._pos),
            "snapshots": int(stats.get("snapshots", len(self._snapshots))),
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=True, sort_keys=True) + "\n")


def _argmax(logits: Any) -> int:
    return int(np.asarray(logits, dtype=np.float32).reshape(-1).argmax())


def _common_prefix(cached: list[int], prompt: list[int]) -> int:
    limit = min(len(cached), len(prompt))
    index = 0
    while index < limit and cached[index] == prompt[index]:
        index += 1
    return index


def _raise_template_error(message: str) -> None:
    raise ValueError(message)


def _swap_used_bytes() -> int | None:
    try:
        output = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                                capture_output=True, text=True, timeout=2.0)
    except Exception:
        return None
    for part in output.stdout.split():
        if part.endswith("M") or part.endswith("G"):
            try:
                value = float(part[:-1])
            except ValueError:
                continue
            return int(value * (1024**3 if part.endswith("G") else 1024**2))
    return None


class ByteOutbox:
    """Bound output bytes and reserve room for control and terminal events."""

    def __init__(self, sink: Callable[[bytes], None], limit: int = 4 * MAX_FRAME_BYTES):
        self.sink = sink
        self.limit = limit
        self.pending_bytes = 0
        self.items: deque[bytes] = deque()
        self.condition = threading.Condition()
        self.failure: BaseException | None = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name="apxinf-output", daemon=True)
        self.thread.start()

    def put(self, frame: dict[str, Any], critical: bool = False) -> None:
        payload = contracts.encode_frame(frame)
        with self.condition:
            if self.failure or self.closed:
                raise ChannelFailure("The output channel is closed.")
            ceiling = self.limit if critical else self.limit - MAX_FRAME_BYTES
            if self.pending_bytes + len(payload) > ceiling:
                if critical:
                    raise ChannelFailure("The reserved output capacity is exhausted.")
                raise WorkerFailure("slow_consumer", "The output consumer cannot keep pace.")
            self.items.append(payload)
            self.pending_bytes += len(payload)
            self.condition.notify()

    def _write(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.items or self.closed)
                if not self.items:
                    return
                payload = self.items.popleft()
            try:
                self.sink(payload)
            except BaseException as error:
                with self.condition:
                    self.failure = error
                    self.condition.notify_all()
                return
            with self.condition:
                self.pending_bytes -= len(payload)
                self.condition.notify_all()

    def close(self, timeout: float = 2.0) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive() or self.failure:
            raise ChannelFailure("The output channel did not finish.")


@dataclass
class Attempt:
    command: dict[str, Any]
    started_ns: int
    event_seq: int = 0
    output_count: int = 0
    first_token_ns: int = 0
    stop_status: str | None = None
    stop_cause: str | None = None
    terminal: bool = False
    payload_digest: str = ""

    def __post_init__(self) -> None:
        content = {key: value for key, value in self.command.items() if key != "command_id"}
        self.payload_digest = hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def key(self) -> tuple[str, int]:
        return (self.command["request_id"], self.command["attempt"])


class TextWorker:
    """Serialize model work and terminal decisions on one execution owner."""

    def __init__(
        self,
        runtime: Any,
        worker_epoch: str,
        emit: Callable[..., None],
        *,
        max_output_tokens: int = 2048,
        output_batch_tokens: int = 1,
    ):
        self.runtime = runtime
        self.worker_epoch = worker_epoch
        self.emit = emit
        self.max_output_tokens = max_output_tokens
        self.output_batch_tokens = output_batch_tokens
        self.commands: queue.Queue = queue.Queue(maxsize=8)
        self.controls: queue.Queue = queue.Queue(maxsize=32)
        self.input_done = threading.Event()
        self.input_fault: BaseException | None = None
        self.active: Attempt | None = None
        self.attempts: dict[tuple[str, int], Attempt] = {}
        self.command_records: dict[str, tuple[bytes, dict[str, Any] | None]] = {}
        self.completed_requests = 0
        self.cancelled_requests = 0
        self.draining = False
        self.shutdown_requested = False
        self.drained_sent = False
        self.capabilities = {
            "text": True, "image": False, "greedy": True, "sampling": False,
            "token_events": True, "cancel_granularity": "token_or_prefill_chunk",
            "max_active_sequences": 1, "max_context": runtime.max_context,
            "max_output_tokens": max_output_tokens, "exact_append": False,
            "state_schema": "none",
        }
        self.model_revision = contracts.canonical_identity_digest(
            "apxinf-model-identity-v1", runtime.manifest
        )
        self.capability_revision = contracts.canonical_identity_digest(
            "apxinf-capability-v1", self.capabilities
        )

    def frame(self, kind: str, **fields: Any) -> dict[str, Any]:
        return {"protocol": "apxinf-worker/2.0", "kind": kind,
                "worker_epoch": self.worker_epoch, **fields}

    @staticmethod
    def error(code: str, message: str, scope: str = "request") -> dict[str, str]:
        return {"code": code, "message": message, "scope": scope, "state_validity": "none"}

    def ready(self) -> dict[str, Any]:
        return self.frame(
            "ready", model_revision=self.model_revision,
            capability_revision=self.capability_revision,
            model_manifest=self.runtime.manifest, capabilities=self.capabilities,
            limits={"max_frame_bytes": MAX_FRAME_BYTES, "max_depth": 16,
                    "max_commands": MAX_COMMANDS},
            model_path=self.runtime.model_path, eos_token_ids=self.runtime.eos_token_ids,
            vocab_size=self.runtime.vocab_size, runtime=self.runtime.runtime,
            memory=self.runtime.memory(),
        )

    def read_commands(self, source: Any) -> None:
        """Read bytes independently. Do not access the runtime from this thread."""
        try:
            while line := source.readline(MAX_FRAME_BYTES + 1):
                if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
                    raise ChannelFailure("The input frame is too large or incomplete.")
                invalid = False
                try:
                    command = contracts.decode_frame(line)
                except contracts.ContractError as error:
                    if error.frame is None:
                        raise
                    contracts.validate_command_envelope(error.frame)
                    command = error.frame
                    invalid = True
                if command["worker_epoch"] != self.worker_epoch:
                    raise ChannelFailure("The command has a different worker epoch.")
                if command["kind"] not in {
                    "submit", "prepare_input", "cancel_request", "stop_generation",
                    "query_stats", "drain", "shutdown",
                }:
                    raise ChannelFailure("The worker received an event instead of a command.")
                selected = self.commands if command["kind"] in {"submit", "prepare_input"} else self.controls
                try:
                    selected.put_nowait(InvalidCommand(command) if invalid else command)
                except queue.Full as error:
                    raise ChannelFailure("The command mailbox is full.") from error
                if command["kind"] == "shutdown" and not invalid:
                    # A shutdown ends input. Do not hold stdin's buffered lock at exit.
                    break
        except BaseException as error:
            self.input_fault = error
        finally:
            self.input_done.set()

    def _send(self, frame: dict[str, Any], critical: bool = True) -> None:
        self.emit(frame, critical=critical)

    def _event(self, attempt: Attempt, kind: str, **fields: Any) -> None:
        frame = self.frame(kind, request_id=attempt.key[0], attempt=attempt.key[1],
                           event_seq=attempt.event_seq, **fields)
        self._send(frame, critical=kind not in {"tokens", "prefill_progress"})
        attempt.event_seq += 1

    def _reply(self, command: dict[str, Any], status: str, error: dict | None = None) -> None:
        fields = {"command_id": command["command_id"], "status": status}
        if error is not None:
            fields["error"] = error
        frame = self.frame("command_result", **fields)
        self._send(frame)
        record = self.command_records.get(command["command_id"])
        if record is not None:
            self.command_records[command["command_id"]] = (record[0], frame)

    def _register_command(self, command: dict[str, Any]) -> bool:
        identity = hashlib.sha256(json.dumps(
            command, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()).digest()
        previous = self.command_records.get(command["command_id"])
        if previous is not None:
            if previous[0] != identity:
                self._send(self.frame("command_result", command_id=command["command_id"],
                                     status="error", error=self.error(
                                         "invalid_request", "The command ID has conflicting content.")))
            elif command["kind"] in {"cancel_request", "stop_generation"} and (
                registered := self.attempts.get((command["request_id"], command["attempt"]))
            ) is not None and registered.terminal:
                self._reply(command, "already_terminal")
            elif previous[1] is not None:
                self._send(previous[1])
            else:
                self._reply(command, "already_registered")
            return False
        if len(self.command_records) >= MAX_COMMANDS:
            raise ChannelFailure("The command history limit requires a new worker epoch.")
        self.command_records[command["command_id"]] = (identity, None)
        return True

    def _control(self, command: dict[str, Any]) -> None:
        if not self._register_command(command):
            return
        kind = command["kind"]
        if kind in {"drain", "shutdown"}:
            self.draining = True
            self.shutdown_requested |= kind == "shutdown"
            self._reply(command, "accepted")
            return
        if kind == "query_stats":
            frame = self.frame("stats", command_id=command["command_id"],
                               active_requests=int(self.active is not None),
                               completed_requests=self.completed_requests,
                               cancelled_requests=self.cancelled_requests)
            self._send(frame)
            old = self.command_records[command["command_id"]]
            self.command_records[command["command_id"]] = (old[0], frame)
            return
        attempt = self.attempts.get((command["request_id"], command["attempt"]))
        if attempt is None:
            self._reply(command, "error", self.error("invalid_request", "The attempt is not registered."))
        elif attempt.terminal:
            self._reply(command, "already_terminal")
        elif kind == "stop_generation" and command["output_token_count"] > attempt.output_count:
            self._reply(command, "error", self.error("invalid_request", "The output boundary is not available."))
        else:
            if attempt.stop_status is None:
                if kind == "stop_generation":
                    attempt.stop_status, attempt.stop_cause = "completed", command["cause"]
                elif command["reason"] == "deadline_exceeded":
                    attempt.stop_status, attempt.stop_cause = "expired", "deadline_exceeded"
                else:
                    attempt.stop_status, attempt.stop_cause = "cancelled", command["reason"]
            self._reply(command, "accepted")

    def _check_controls(self, attempt: Attempt | None = None) -> None:
        if self.input_fault is not None:
            raise ChannelFailure("The input channel failed validation.")
        while True:
            try:
                command = self.controls.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, InvalidCommand):
                self._invalid(command.frame)
            else:
                self._control(command)
        if attempt is None:
            return
        elapsed = time.monotonic_ns() - attempt.started_ns
        if attempt.stop_status is None and elapsed >= attempt.command["remaining_timeout_ms"] * 1_000_000:
            attempt.stop_status, attempt.stop_cause = "expired", "deadline_exceeded"
        if attempt.stop_status is not None:
            raise GenerationStopped()

    def _prepare(self, command: dict[str, Any]) -> None:
        if command["model_revision"] != self.model_revision:
            self._reply(command, "error", self.error("invalid_request", "The model revision does not match."))
            return
        try:
            tokens = self.runtime.prepare(command)
            if not tokens or len(tokens) > self.runtime.max_context:
                raise WorkerFailure("context_limit", "The prepared input exceeds the context limit.")
            if any(type(token) is not int or not 0 <= token < self.runtime.vocab_size for token in tokens):
                raise WorkerFailure("internal_error", "The tokenizer returned an invalid token.")
            frame = self.frame(
                "prepared_input", command_id=command["command_id"],
                model_revision=self.model_revision, token_ids=tokens,
                effective_input_tokens=len(tokens), prompt_digest=contracts.token_prefix_digest(tokens),
            )
            self._send(frame)
            # Retain the digest, not a second copy of a potentially large prompt.
        except WorkerFailure as error:
            self._reply(command, "error", self.error(error.code, str(error)))
        except Exception:
            self._reply(command, "error", self.error("invalid_request", "The model template rejected the input."))

    def _submit_error(self, command: dict[str, Any]) -> dict | None:
        if self.draining:
            return self.error("model_unavailable", "The worker is draining.")
        if command["model_revision"] != self.model_revision or command["capability_revision"] != self.capability_revision:
            return self.error("invalid_request", "The request revisions do not match the worker.")
        if command["max_tokens"] > self.max_output_tokens:
            return self.error("unsupported_feature", "The requested output exceeds the deployment limit.")
        if len(command["token_ids"]) + command["max_tokens"] > self.runtime.max_context:
            return self.error("context_limit", "The input and output allowance exceed the context limit.")
        if any(token >= self.runtime.vocab_size for token in command["token_ids"] + command["eos_token_ids"]):
            return self.error("invalid_request", "A token exceeds the model vocabulary.")
        return None

    def _submit(self, command: dict[str, Any]) -> None:
        key = (command["request_id"], command["attempt"])
        previous = self.attempts.get(key)
        if previous is not None:
            if previous.payload_digest == Attempt(command, 0).payload_digest:
                self._reply(command, "already_registered")
            else:
                self._reply(command, "error", self.error("invalid_request", "The attempt has conflicting content."))
            return
        attempt = Attempt(command, time.monotonic_ns())
        self.attempts[key] = attempt
        error = self._submit_error(command)
        if error:
            self._event(attempt, "rejected", error=error)
            self._finish(attempt, "failed", error["code"], error, mutated=False)
            return
        self.active = attempt
        self._event(attempt, "accepted")
        status, cause = "completed", "length"
        error = None
        pending_tokens: list[int] = []
        pending_text: list[str] = []
        generator = None
        mutated = False
        decoder_valid = True
        output_rejected = False

        def flush() -> None:
            nonlocal output_rejected
            if pending_tokens:
                try:
                    self._event(attempt, "tokens", output_index=attempt.output_count,
                                token_ids=list(pending_tokens), text_delta="".join(pending_text))
                except Exception:
                    output_rejected = True
                    raise
                attempt.output_count += len(pending_tokens)
                pending_tokens.clear()
                pending_text.clear()

        last_progress = -1

        def progress(consumed: int) -> None:
            nonlocal last_progress
            self._check_controls(attempt)
            if consumed > last_progress:
                self._event(attempt, "prefill_progress", consumed_tokens=consumed)
                last_progress = consumed

        try:
            self._check_controls(attempt)
            if command["max_tokens"]:
                generator = self.runtime.generate(command, progress)
                mutated = True
                for token in generator:
                    if not 0 <= token < self.runtime.vocab_size:
                        raise WorkerFailure("internal_error", "The model returned an invalid token.")
                    if not attempt.first_token_ns:
                        attempt.first_token_ns = time.monotonic_ns()
                    eos = token in command["eos_token_ids"]
                    try:
                        text = self.runtime.decode(token, eos=eos)
                    except Exception:
                        decoder_valid = False
                        raise
                    pending_tokens.append(token)
                    pending_text.append(text)
                    if len(pending_tokens) >= self.output_batch_tokens or attempt.output_count == 0 or eos:
                        flush()
                    self._check_controls(attempt)
                    if eos:
                        cause = "eos"
                        break
                flush()
                if cause == "length" and attempt.output_count != command["max_tokens"]:
                    raise WorkerFailure("internal_error", "Generation ended before the output allowance.")
        except GenerationStopped:
            status, cause = attempt.stop_status, attempt.stop_cause
            try:
                flush()
            except WorkerFailure:
                status, cause = "cancelled", "slow_consumer"
        except WorkerFailure as failure:
            if failure.code == "slow_consumer":
                status, cause = "cancelled", "slow_consumer"
            else:
                status, cause = "failed", failure.code
                error = self.error(failure.code, str(failure))
        except ChannelFailure:
            raise
        except Exception:
            traceback.print_exc(file=sys.stderr)
            status, cause = "failed", "internal_error"
            error = self.error("internal_error", "Model execution failed.")
        finally:
            if generator is not None:
                generator.close()
        if status == "failed" and not output_rejected:
            try:
                flush()
            except ChannelFailure:
                raise
            except Exception:
                # An output rejection cannot replace the observed execution failure.
                pass
        self._finish(attempt, status, cause, error, mutated=mutated,
                     finalize_text=decoder_valid and not output_rejected)
        self.active = None

    def _finish(self, attempt: Attempt, status: str, cause: str,
                error: dict | None, *, mutated: bool, finalize_text: bool = True) -> None:
        tail = ""
        consumed = 0
        peak = 0
        if mutated:
            try:
                tail = self.runtime.finish_text() if finalize_text and cause != "slow_consumer" else ""
                consumed = self.runtime.consumed_position
                peak = self.runtime.memory()["peak_bytes"]
            except Exception:
                status, cause = "failed", "internal_error"
                error = self.error("internal_error", "The output finalization failed.")
        history = len(attempt.command["token_ids"]) + attempt.output_count
        # An aborted internal lookahead cannot establish reusable state.
        if consumed > history:
            consumed = 0
        now = time.monotonic_ns()
        fields = {
            "status": status, "cause": cause,
            "usage": {"input_tokens": len(attempt.command["token_ids"]), "output_tokens": attempt.output_count},
            "state_result": {"validity": "none", "consumed_position": consumed, "history_count": history},
            "text_delta": tail,
            "metrics": {"elapsed_ns": now - attempt.started_ns,
                        "ttft_ns": (attempt.first_token_ns - attempt.started_ns) if attempt.output_count else 0,
                        "peak_memory_bytes": peak},
        }
        if error is not None:
            fields["error"] = error
        try:
            self.runtime.write_ledger(attempt, status, cause, fields["usage"],
                                      fields["metrics"], executed=mutated)
        except Exception as error:
            print(f"ledger write failed: {error!r}", file=sys.stderr)
        self._event(attempt, "terminal", **fields)
        attempt.terminal = True
        if status == "completed":
            self.completed_requests += 1
        if status == "cancelled":
            self.cancelled_requests += 1
        if mutated:
            try:
                self.runtime.settle()
            except Exception as failure:
                # Do not return leases when device settlement did not complete.
                raise ChannelFailure("Device resource settlement failed.") from failure
        self._event(attempt, "resources_released",
                    released_lease_ids=attempt.command["capacity_lease_ids"], retained_lease_ids=[])
        # Keep only duplicate-detection and terminal metadata within the epoch.
        attempt.command = {"request_id": attempt.key[0], "attempt": attempt.key[1]}

    def _invalid(self, command: dict[str, Any]) -> None:
        if not self._register_command(command):
            return
        error = self.error("invalid_request", "The command content does not match the profile.")
        if command["kind"] != "submit":
            self._reply(command, "error", error)
            return
        key = (command["request_id"], command["attempt"])
        if key in self.attempts:
            self._reply(command, "error", error)
            return
        safe_command = dict(command)
        tokens = command.get("token_ids")
        safe_command["token_ids"] = tokens if type(tokens) is list else []
        attempt = Attempt(safe_command, time.monotonic_ns())
        self.attempts[key] = attempt
        self._event(attempt, "rejected", error=error)
        self._finish(attempt, "failed", "invalid_request", error, mutated=False)

    def handle(self, command: dict[str, Any] | InvalidCommand) -> None:
        if isinstance(command, InvalidCommand):
            self._invalid(command.frame)
            return
        contracts.validate_frame(command)
        if command["worker_epoch"] != self.worker_epoch:
            raise ChannelFailure("The command has a different worker epoch.")
        if command["kind"] not in {"submit", "prepare_input"}:
            self._control(command)
            return
        if not self._register_command(command):
            return
        if command["kind"] == "prepare_input":
            self._prepare(command)
        else:
            self._submit(command)

    def run(self, source: Any) -> None:
        self._send(self.ready())
        reader = threading.Thread(target=self.read_commands, args=(source,), name="apxinf-input", daemon=True)
        reader.start()
        try:
            while True:
                # Register a queued submit before applying its later control.
                try:
                    command = self.commands.get(timeout=0.01)
                except queue.Empty:
                    command = None
                if command is not None:
                    self.handle(command)
                self._check_controls()
                if self.draining and not self.drained_sent and self.commands.empty():
                    self._send(self.frame("drained"))
                    self.drained_sent = True
                if (self.shutdown_requested or self.input_done.is_set()) and self.commands.empty():
                    break
        except ChannelFailure:
            # Process exit fences attempts whose resources remain unsettled.
            with contextlib.suppress(Exception):
                self._send(self.frame("worker_fault", error=self.error(
                    "protocol_fault", "The worker channel cannot continue.", "worker")))
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--protocol", default="apxinf-worker/2.0")
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--output-batch-tokens", type=int, default=1)
    parser.add_argument("--memory-limit-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="on")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--engine-root", default=None)
    parser.add_argument("--snapshot-interval", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.protocol != "apxinf-worker/2.0":
        parser.error("The protocol version is not supported.")
    try:
        contracts.validate_frame({"protocol": arguments.protocol, "kind": "drained",
                                  "worker_epoch": arguments.worker_epoch})
    except contracts.ContractError:
        parser.error("The worker epoch is invalid.")
    if not 1 <= arguments.prefill_step_size <= 8192 or not 1 <= arguments.output_batch_tokens <= 256:
        parser.error("The prefill or output batch size is invalid.")
    if not 1 <= arguments.max_context <= 131072 or not 0 <= arguments.max_output_tokens <= 65536:
        parser.error("The deployment token limits are invalid.")
    if not 1 <= arguments.snapshot_interval <= 8192:
        parser.error("The snapshot interval is invalid.")
    if not 1 <= arguments.memory_limit_bytes <= 9007199254740991:
        parser.error("The memory limit is invalid.")
    output_fd = sys.stdout.fileno()

    def write_all(payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(output_fd, view)
            view = view[written:]

    outbox = ByteOutbox(write_all)
    try:
        runtime = LFM3Runtime(arguments.model, arguments.max_context, arguments.prefill_step_size,
                              arguments.output_batch_tokens, arguments.memory_limit_bytes,
                              prefix_cache=arguments.prefix_cache == "on",
                              ledger_path=arguments.ledger,
                              engine_root=arguments.engine_root,
                              snapshot_interval=arguments.snapshot_interval)
        worker = TextWorker(runtime, arguments.worker_epoch, outbox.put,
                            max_output_tokens=arguments.max_output_tokens,
                            output_batch_tokens=arguments.output_batch_tokens)
        worker.run(sys.stdin.buffer)
        outbox.close()
        return 0
    except Exception as error:
        print(f"Text worker stopped: {type(error).__name__}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

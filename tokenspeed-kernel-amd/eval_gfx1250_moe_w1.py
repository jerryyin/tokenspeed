#!/usr/bin/env python3
"""Evaluation-only H0 versus all-N W1 driver for the gfx1250 decode kernel.

Device modes intentionally expose one tuple only: BM16/BN128/BK512/8 warps/
3 buffers.  ``--self-test`` is stdlib-only and must pass before a gated device
run.  This is a W1 dispatch-boundary diagnostic, not a production-decode
denominator.
"""

from __future__ import annotations

import argparse
import ast
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path("/data/lock/amd-gpu.lock")
BREADCRUMB_PATH = Path("/root/ablation-breadcrumb.jsonl")
PREFLIGHT_PATH = Path("/root/PREFLIGHT.txt")
MCE_PATH = Path("/root/MCE.json")

BLOCK_M = 16
BLOCK_N = 128
BLOCK_K = 512
NUM_WARPS = 8
NUM_BUFFERS = 3
NUM_EXPERTS = 256
TOP_K = 4
K = 5120
N = 10240
BPE_TO_B = {1: 64, 4: 256, 16: 1024}
ARMS = {"H0": False, "D": True}

EXPECTED_BOOT_PREFIX = "d561b228"
EXPECTED_TORCH = "2.10.0+rocm7.13.0a20260505-a0"
EXPECTED_TOKENSPEED_TRITON = "3.8.10.post20260721"
EXPECTED_TOKENSPEED_PROTON = "3.8.10.post20260721"
EXPECTED_OFFICIAL_CORRECTNESS_HEAD = "219de4f668423aab677e4e5d114371fcec6a46be"
MAX_PREFLIGHT_AGE_SECONDS = 90 * 60
TIMING_PAIRS = 6
TIMING_WARMUPS = 10

SOURCE_FILES = (
    "tokenspeed-kernel-amd/eval_gfx1250_moe_w1.py",
    "tokenspeed-kernel/test/ops/moe/test_gluon_mxfp4_amd.py",
    "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/moe/mxfp4/_common.py",
    "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/moe/mxfp4/decode.py",
    "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/moe/mxfp4/fused.py",
    "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/moe/mxfp4/weight_preprocess.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_fsynced_jsonl(path: Path, record: dict[str, Any]) -> None:
    encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def tflops(time_us: float, m: int) -> float:
    time_ms = time_us / 1000.0
    return 2.0 * m * N * K / (time_ms * 1.0e-3) / 1.0e12


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _timing_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_bpe: dict[str, Any] = {}
    for bpe in sorted({int(sample["batch_per_expt"]) for sample in samples}):
        cell = [sample for sample in samples if sample["batch_per_expt"] == bpe]
        pairs: dict[int, dict[str, dict[str, Any]]] = {}
        for sample in cell:
            pairs.setdefault(int(sample["pair_index"]), {})[sample["arm"]] = sample
        effects = []
        h0_first = []
        d_first = []
        for pair_index in sorted(pairs):
            pair = pairs[pair_index]
            if set(pair) != set(ARMS):
                raise ValueError(f"incomplete pair {pair_index} at BPE={bpe}")
            effect = (
                100.0
                * (pair["H0"]["time_us"] - pair["D"]["time_us"])
                / pair["H0"]["time_us"]
            )
            effects.append(effect)
            first = min(pair.values(), key=lambda item: item["order_index"])["arm"]
            (h0_first if first == "H0" else d_first).append(effect)
        by_bpe[str(bpe)] = {
            "pairs": len(pairs),
            "median_h0_us": _median(
                [float(item["time_us"]) for item in cell if item["arm"] == "H0"]
            ),
            "median_d_us": _median(
                [float(item["time_us"]) for item in cell if item["arm"] == "D"]
            ),
            "median_h0_tflops": _median(
                [float(item["tflops"]) for item in cell if item["arm"] == "H0"]
            ),
            "median_d_tflops": _median(
                [float(item["tflops"]) for item in cell if item["arm"] == "D"]
            ),
            "median_d_speedup_pct": _median(effects),
            "median_d_speedup_pct_h0_first": _median(h0_first),
            "median_d_speedup_pct_d_first": _median(d_first),
            "order_signs_agree": bool(h0_first and d_first)
            and math.copysign(1.0, statistics.median(h0_first))
            == math.copysign(1.0, statistics.median(d_first)),
        }
    return by_bpe


def build_record(
    *,
    mode: str,
    run_id: str,
    batch_per_expt: list[int],
    pairs: int,
    warmups: int,
    runtime: dict[str, Any],
    source_hashes: dict[str, str],
    samples: list[dict[str, Any]],
    correctness: list[dict[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "campaign": "a8w4-decode-performance",
        "instrument": "tokenspeed-gfx1250-direct-w1",
        "evidence_scope": "authorized W1 dispatch-boundary diagnostic; not full production decode",
        "mode": mode,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime": runtime,
        "source_sha256": source_hashes,
        "tuple": {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "NUM_WARPS": NUM_WARPS,
            "NUM_BUFFERS": NUM_BUFFERS,
            "K": K,
            "N": N,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "scale_load_mode": "swizzle",
            "output_dtype": "bfloat16",
        },
        "arms": {
            "H0": {"all_n_layout": False, "description": "stock warp bases"},
            "D": {"all_n_layout": True, "description": "all-N warp bases"},
        },
        "batch_per_expt": list(batch_per_expt),
        "pairs_per_shape": pairs,
        "warmups_per_arm_per_shape": warmups,
        "correctness": correctness,
        "samples": samples,
        "launch_contract": {
            "selected_launch_cap_per_shape": 2 * pairs,
            "arm_order": "alternating H0,D then D,H0",
            "warmup_timing_path_matches_selected_timing_path": True,
            "correctness_before_timing": True,
            "correctness_after_timing": True,
        },
        "what_this_can_establish": (
            "same-process H0-versus-all-N W1 latency at TokenSpeed's swizzled-scale "
            "dispatch boundary and the exactly pinned tuple"
        ),
        "what_this_cannot_establish": (
            "full-decode speedup, strided-scale parity with the campaign harness, or W2/combine behavior"
        ),
    }
    record["summary"] = _timing_summary(samples) if mode == "timing" else {}
    return record


def _self_test() -> int:
    if "torch" in sys.modules or "tokenspeed_triton" in sys.modules:
        raise AssertionError("device packages imported before off-hardware self-test")
    if (BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_BUFFERS) != (16, 128, 512, 8, 3):
        raise AssertionError("evaluation tuple drift")
    if ARMS != {"H0": False, "D": True}:
        raise AssertionError("arm mapping drift")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("prepare_persistent_state", "build_persistent_state"):
        if forbidden in referenced_names:
            raise AssertionError(f"forbidden builder reference: {forbidden}")
    test = (REPO_ROOT / SOURCE_FILES[1]).read_text(encoding="utf-8")
    common = (REPO_ROOT / SOURCE_FILES[2]).read_text(encoding="utf-8")
    decode = (REPO_ROOT / SOURCE_FILES[3]).read_text(encoding="utf-8")
    fused = (REPO_ROOT / SOURCE_FILES[4]).read_text(encoding="utf-8")
    required_fragments = (
        "elif num_warps == 4:",
        "[tiles_per_warp, 0]",
        "while n_stride < tiles_per_warp * num_warps:",
        "ALL_N_LAYOUT: gl.constexpr = True",
        "ALL_N_LAYOUT=all_n_layout",
    )
    joined = "\n".join((test, common, decode, fused))
    missing = [fragment for fragment in required_fragments if fragment not in joined]
    if missing:
        raise AssertionError(f"layout selector propagation missing: {missing}")

    synthetic = []
    for pair_index, order in enumerate((("H0", "D"), ("D", "H0"))):
        for order_index, arm in enumerate(order):
            elapsed = 100.0 if arm == "H0" else 90.0
            synthetic.append(
                {
                    "batch_per_expt": 1,
                    "B": 64,
                    "M": 256,
                    "pair_index": pair_index,
                    "order_index": order_index,
                    "arm": arm,
                    "all_n_layout": ARMS[arm],
                    "time_us": elapsed,
                    "tflops": tflops(elapsed, 256),
                }
            )
    runtime = {
        "off_hardware": True,
        "device_launches": 0,
        "gpu_lock_acquisitions": 0,
        "torch_imported": False,
    }
    hashes = {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_FILES}
    record = build_record(
        mode="timing",
        run_id="self-test",
        batch_per_expt=[1],
        pairs=2,
        warmups=1,
        runtime=runtime,
        source_hashes=hashes,
        samples=synthetic,
        correctness=[{"batch_per_expt": 1, "bitwise_equal": True}],
        started_at_utc="2026-08-25T00:00:00Z",
        completed_at_utc="2026-08-25T00:00:01Z",
    )
    with tempfile.TemporaryDirectory(prefix="tokenspeed-w1-selftest-") as directory:
        output = Path(directory) / "record.json"
        atomic_write_json(output, record)
        round_trip = json.loads(output.read_text(encoding="utf-8"))
    if round_trip != record:
        raise AssertionError("complete record atomic JSON round-trip mismatch")
    if "torch" in sys.modules or "tokenspeed_triton" in sys.modules:
        raise AssertionError("off-hardware self-test initialized a device package")
    print(
        json.dumps(
            {
                "status": "PASS",
                "complete_record_round_trip": True,
                "device_launches": 0,
                "gpu_lock_acquisitions": 0,
                "tuple": record["tuple"],
                "arms": record["arms"],
                "source_sha256": hashes,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("PREFLIGHT is absent")
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    if "DEVICE_WORK: ALLOWED" not in text.splitlines():
        raise RuntimeError("PREFLIGHT does not say DEVICE_WORK: ALLOWED")
    fields = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    checked = datetime.fromisoformat(fields["checked_at_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if age < 0 or age > MAX_PREFLIGHT_AGE_SECONDS:
        raise RuntimeError(f"PREFLIGHT is stale: age_seconds={age:.1f}")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if fields.get("boot_id") != boot_id:
        raise RuntimeError("PREFLIGHT boot does not match the running boot")
    if not boot_id.startswith(EXPECTED_BOOT_PREFIX):
        raise RuntimeError(f"unexpected boot: {boot_id}")
    return {
        "sha256": sha256_file(PREFLIGHT_PATH),
        "checked_at_utc": fields["checked_at_utc"],
        "age_seconds": age,
        "boot_id": boot_id,
    }


def _require_external_lock() -> None:
    if os.environ.get("TOKENSPEED_GPU_LOCK_HELD") != str(LOCK_PATH):
        raise RuntimeError(
            f"TOKENSPEED_GPU_LOCK_HELD must equal {LOCK_PATH}; acquire it externally"
        )
    fd = os.open(LOCK_PATH, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            raise RuntimeError("GPU lock marker is set but the lock is not held")
    finally:
        os.close(fd)


def _direct_idle_gate() -> dict[str, Any]:
    kfd_root = Path("/sys/class/kfd/kfd/proc")
    if not kfd_root.is_dir():
        raise RuntimeError(f"KFD process directory is absent: {kfd_root}")
    kfd_pids = sorted(path.name for path in kfd_root.iterdir())
    if kfd_pids:
        raise RuntimeError(f"direct idle gate found KFD PIDs: {kfd_pids}")
    rocm_smi = next(
        (
            candidate
            for candidate in (
                Path("/opt/venv/bin/rocm-smi"),
                Path("/opt/rocm/bin/rocm-smi"),
                Path("/usr/bin/rocm-smi"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if rocm_smi is None:
        raise RuntimeError("rocm-smi is absent")
    pid_command = [str(rocm_smi), "--showpids"]
    memory_command = [str(rocm_smi), "--showmemuse"]
    pid_output = subprocess.check_output(
        pid_command, text=True, stderr=subprocess.STDOUT, timeout=20
    )
    memory_output = subprocess.check_output(
        memory_command, text=True, stderr=subprocess.STDOUT, timeout=20
    )
    percentages = [
        int(value)
        for value in re.findall(
            r"GPU Memory Allocated \(VRAM%\).*?([0-9]+)", memory_output
        )
    ]
    if "No KFD PIDs currently running" not in pid_output:
        raise RuntimeError("rocm-smi --showpids did not prove zero KFD PIDs")
    if not percentages or any(percentages):
        raise RuntimeError(
            "rocm-smi --showmemuse did not prove zero VRAM allocation: "
            f"vram_allocated_percent={percentages}"
        )
    return {
        "exclusive_lock": str(LOCK_PATH),
        "sysfs_kfd_pids": kfd_pids,
        "kfd_pid_count": len(kfd_pids),
        "vram_allocated_percent": percentages,
        "gpu_use_percent_considered": False,
        "rocm_smi_showpids_command": pid_command,
        "rocm_smi_showpids": pid_output,
        "rocm_smi_showmemuse_command": memory_command,
        "rocm_smi_showmemuse": memory_output,
    }


def _require_device_envelope(expected_head: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if not (2 <= now.hour < 14):
        raise RuntimeError(
            f"outside the 02:00-14:00 UTC device window: {now.isoformat()}"
        )
    _require_external_lock()
    preflight = _parse_preflight()
    idle_gate = _direct_idle_gate()
    head = git_output("rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(f"TokenSpeed HEAD mismatch: {head} != {expected_head}")
    if git_output("status", "--porcelain"):
        raise RuntimeError("TokenSpeed worktree must be clean for device work")
    mce = None
    if MCE_PATH.is_file():
        mce = json.loads(MCE_PATH.read_text(encoding="utf-8"))
        verdict = str(mce.get("tripwire_verdict", mce.get("verdict", ""))).upper()
        if verdict == "TRIPPED":
            raise RuntimeError("MCE tripwire is TRIPPED")
    return {
        "preflight": preflight,
        "idle_gate": idle_gate,
        "git_head": head,
        "mce_start": mce,
    }


def _official_correctness_gate(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"official correctness result is absent: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "official correctness result hash drift: "
            f"{observed_sha256} != {expected_sha256}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    tokenspeed = document.get("tokenspeed", {})
    required = {
        "status": document.get("status"),
        "returncode": document.get("returncode"),
        "reported_passed_test_cases": document.get("reported_passed_test_cases"),
        "expected_test_cases": document.get("expected_test_cases"),
        "selected_timing_launches": document.get("selected_timing_launches"),
        "tokenspeed_head": tokenspeed.get("head"),
    }
    expected = {
        "status": "PASS",
        "returncode": 0,
        "reported_passed_test_cases": 6,
        "expected_test_cases": 6,
        "selected_timing_launches": 0,
        "tokenspeed_head": EXPECTED_OFFICIAL_CORRECTNESS_HEAD,
    }
    if required != expected:
        raise RuntimeError(
            "official TokenSpeed correctness prerequisite did not pass exactly: "
            f"observed={required}, expected={expected}"
        )
    return {"path": str(path.resolve()), "sha256": observed_sha256, **required}


class Breadcrumbs:
    def __init__(self, run_id: str, command: list[str], boot_id: str):
        self.run_id = run_id
        self.command = command
        self.boot_id = boot_id
        self.ordinal = 0

    def call(
        self,
        *,
        phase: str,
        bpe: int,
        arm: str,
        selected: bool,
        function: Callable[[], Any],
    ) -> Any:
        self.ordinal += 1
        launch_id = f"{self.run_id}:{self.ordinal}:{uuid.uuid4().hex}"
        common = {
            "schema_version": 1,
            "record_type": "tokenspeed_w1_launch",
            "run_id": self.run_id,
            "launch_id": launch_id,
            "boot_id": self.boot_id,
            "phase": phase,
            "launch_ordinal": self.ordinal,
            "selected_timing_launch": selected,
            "target": arm,
            "config": {
                "batch_per_expt": bpe,
                "B": BPE_TO_B[bpe],
                "M": BPE_TO_B[bpe] * TOP_K,
                "N": N,
                "K": K,
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "NUM_WARPS": NUM_WARPS,
                "NUM_BUFFERS": NUM_BUFFERS,
                "all_n_layout": ARMS[arm],
            },
            "command": self.command,
        }
        append_fsynced_jsonl(
            BREADCRUMB_PATH,
            {**common, "event": "before", "recorded_at_utc": utc_now()},
        )
        try:
            result = function()
        except BaseException as error:
            append_fsynced_jsonl(
                BREADCRUMB_PATH,
                {
                    **common,
                    "event": "outcome",
                    "recorded_at_utc": utc_now(),
                    "status": "exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            )
            raise
        append_fsynced_jsonl(
            BREADCRUMB_PATH,
            {
                **common,
                "event": "outcome",
                "recorded_at_utc": utc_now(),
                "status": "returned",
            },
        )
        return result


def _device_main(args: argparse.Namespace) -> int:
    envelope = _require_device_envelope(args.expected_head)
    official_correctness = _official_correctness_gate(
        args.official_correctness_result,
        args.official_correctness_sha256,
    )

    import torch
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (
        _precomputed_topk_route,
        gluon_mxfp_dispatch_swiglu,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (
        _swizzle_mxfp4,
    )

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"torch mismatch: {torch.__version__} != {EXPECTED_TORCH}")
    triton_version = importlib.metadata.version("tokenspeed-triton")
    proton_version = importlib.metadata.version("tokenspeed-proton")
    if triton_version != EXPECTED_TOKENSPEED_TRITON:
        raise RuntimeError(f"tokenspeed-triton mismatch: {triton_version}")
    if proton_version != EXPECTED_TOKENSPEED_PROTON:
        raise RuntimeError(f"tokenspeed-proton mismatch: {proton_version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is unavailable")

    started = utc_now()
    source_hashes = {
        relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_FILES
    }
    runtime = {
        **envelope,
        "official_correctness_prerequisite": official_correctness,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "tokenspeed_triton": triton_version,
        "tokenspeed_proton": proton_version,
        "device_name": torch.cuda.get_device_name(0),
        "exclusive_lock": str(LOCK_PATH),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    breadcrumbs = Breadcrumbs(
        args.run_id,
        [str(Path(sys.executable).resolve()), *sys.argv],
        envelope["preflight"]["boot_id"],
    )

    weight_generator = torch.Generator(device="cuda").manual_seed(271828)
    raw_weight = torch.randint(
        0,
        256,
        (NUM_EXPERTS, N, K // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=weight_generator,
    )
    raw_scale = torch.full(
        (NUM_EXPERTS, N, K // 32), 127, dtype=torch.uint8, device="cuda"
    )
    weight, weight_scale = _swizzle_mxfp4(raw_weight, raw_scale)
    del raw_weight, raw_scale
    bias = torch.zeros((NUM_EXPERTS, N), dtype=torch.float32, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    fixtures: dict[int, tuple[Any, Any, Any]] = {}
    for bpe in args.batch_per_expt:
        b = BPE_TO_B[bpe]
        input_generator = torch.Generator(device="cuda").manual_seed(314159 + bpe)
        x = torch.randn(
            (b, K), dtype=torch.bfloat16, device="cuda", generator=input_generator
        ).to(torch.float8_e4m3fn)
        token_ids = torch.arange(b, device="cuda", dtype=torch.int32)[:, None]
        slots = torch.arange(TOP_K, device="cuda", dtype=torch.int32)[None, :]
        topk_ids = (token_ids + 64 * slots) % NUM_EXPERTS
        topk_weights = torch.full((b, TOP_K), 0.25, dtype=torch.float32, device="cuda")
        metadata, gather, _scatter, _gate = _precomputed_topk_route(
            topk_weights, topk_ids, NUM_EXPERTS
        )
        fixtures[bpe] = (x, metadata, gather)
    torch.cuda.synchronize()

    def launch(bpe: int, arm: str):
        x, metadata, gather = fixtures[bpe]
        return gluon_mxfp_dispatch_swiglu(
            x,
            weight,
            weight_scale,
            x_format="e4m3",
            x_global_scale=1.0,
            bias=bias,
            a_ragged_metadata=metadata,
            gather_indx=gather,
            out_dtype=torch.bfloat16,
            swiglu_alpha=1.0,
            swiglu_limit=1.0,
            swiglu_beta=1.0,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
            block_k=BLOCK_K,
            num_warps=NUM_WARPS,
            num_buffers=NUM_BUFFERS,
            scale_load_mode="swizzle",
            w_transpose=True,
            decode=True,
            all_n_layout=ARMS[arm],
        )

    def plain_launch(bpe: int, arm: str, phase: str):
        def invoke():
            output = launch(bpe, arm)
            torch.cuda.synchronize()
            return output

        return breadcrumbs.call(
            phase=phase,
            bpe=bpe,
            arm=arm,
            selected=False,
            function=invoke,
        )

    def timed_launch(bpe: int, arm: str, phase: str, selected: bool):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def invoke():
            start.record()
            output = launch(bpe, arm)
            end.record()
            end.synchronize()
            return output, float(start.elapsed_time(end) * 1000.0)

        return breadcrumbs.call(
            phase=phase,
            bpe=bpe,
            arm=arm,
            selected=selected,
            function=invoke,
        )

    def compare_outputs(bpe: int, h0, d, phase: str) -> dict[str, Any]:
        h0_cpu = h0.cpu()
        d_cpu = d.cpu()
        equal = bool(torch.equal(h0_cpu, d_cpu))
        max_abs = float((h0_cpu.float() - d_cpu.float()).abs().max().item())
        result = {
            "batch_per_expt": bpe,
            "B": BPE_TO_B[bpe],
            "M": BPE_TO_B[bpe] * TOP_K,
            "phase": phase,
            "bitwise_equal": equal,
            "max_abs_diff": max_abs,
        }
        if not equal:
            raise AssertionError(f"H0 and D differ at BPE={bpe}: {result}")
        return result

    correctness: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for bpe in args.batch_per_expt:
        # Compile both constexpr variants before correctness or timing.
        plain_launch(bpe, "H0", "compile_prime")
        plain_launch(bpe, "D", "compile_prime")
        before_h0 = plain_launch(bpe, "H0", "correctness_before")
        before_d = plain_launch(bpe, "D", "correctness_before")
        correctness.append(compare_outputs(bpe, before_h0, before_d, "before"))
        if args.mode == "correctness":
            continue

        for warmup_index in range(args.warmups):
            order = ("H0", "D") if warmup_index % 2 == 0 else ("D", "H0")
            for arm in order:
                timed_launch(bpe, arm, "warmup", selected=False)

        last_outputs = {}
        for pair_index in range(args.pairs):
            order = ("H0", "D") if pair_index % 2 == 0 else ("D", "H0")
            for order_index, arm in enumerate(order):
                output, elapsed_us = timed_launch(bpe, arm, "timing", selected=True)
                last_outputs[arm] = output
                m = BPE_TO_B[bpe] * TOP_K
                samples.append(
                    {
                        "batch_per_expt": bpe,
                        "B": BPE_TO_B[bpe],
                        "M": m,
                        "N": N,
                        "K": K,
                        "pair_index": pair_index,
                        "order_index": order_index,
                        "arm": arm,
                        "all_n_layout": ARMS[arm],
                        "time_us": elapsed_us,
                        "time_ms": elapsed_us / 1000.0,
                        "tflops": tflops(elapsed_us, m),
                    }
                )
        correctness.append(
            compare_outputs(bpe, last_outputs["H0"], last_outputs["D"], "after")
        )

    runtime["mce_end"] = (
        json.loads(MCE_PATH.read_text(encoding="utf-8")) if MCE_PATH.is_file() else None
    )
    record = build_record(
        mode=args.mode,
        run_id=args.run_id,
        batch_per_expt=args.batch_per_expt,
        pairs=args.pairs if args.mode == "timing" else 0,
        warmups=args.warmups if args.mode == "timing" else 0,
        runtime=runtime,
        source_hashes=source_hashes,
        samples=samples,
        correctness=correctness,
        started_at_utc=started,
        completed_at_utc=utc_now(),
    )
    atomic_write_json(args.output, record)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "summary": record["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "timing"))
    parser.add_argument("--batch-per-expt", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--pairs", type=int, default=TIMING_PAIRS)
    parser.add_argument("--warmups", type=int, default=TIMING_WARMUPS)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--official-correctness-result", type=Path)
    parser.add_argument("--official-correctness-sha256")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if (
        args.mode is None
        or args.run_id is None
        or args.output is None
        or args.official_correctness_result is None
        or args.official_correctness_sha256 is None
    ):
        parser.error(
            "device mode requires --mode, --run-id, --output and "
            "--official-correctness-result/--official-correctness-sha256"
        )
    if args.expected_head is None or len(args.expected_head) != 40:
        parser.error("device mode requires a full 40-hex --expected-head")
    if len(args.official_correctness_sha256) != 64:
        parser.error("--official-correctness-sha256 must be a full 64-hex digest")
    try:
        int(args.expected_head, 16)
        int(args.official_correctness_sha256, 16)
    except ValueError:
        parser.error("expected HEAD and correctness SHA-256 must be hexadecimal")
    if len(args.batch_per_expt) != 1:
        parser.error("one configuration is permitted per invocation")
    if sorted(set(args.batch_per_expt)) != sorted(args.batch_per_expt):
        parser.error("--batch-per-expt values must be unique and sorted")
    invalid_bpe = [value for value in args.batch_per_expt if value not in BPE_TO_B]
    if invalid_bpe:
        parser.error(f"unsupported --batch-per-expt values: {invalid_bpe}")
    if args.mode == "timing" and args.pairs != TIMING_PAIRS:
        parser.error(f"timing requires exactly {TIMING_PAIRS} pairs")
    if args.mode == "timing" and args.warmups != TIMING_WARMUPS:
        parser.error(f"timing requires exactly {TIMING_WARMUPS} warmups")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    return _device_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

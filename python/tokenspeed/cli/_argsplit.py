# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Argv splitter for ``ts serve``.

A leading positional argument is treated as the model (vllm-style
``ts serve <model> [flags...]``) and rewritten to ``--model <model>``
before routing.

``--headless`` short-circuits the split entirely (see
``split_headless_argv``): the engine is the only process, so every
remaining flag is a ServerArgs flag and no engine/gateway routing applies.

Routing precedence is top-down. The first matching rule wins:

1. Orchestrator-only flags (consumed, never forwarded)
2. ``--model`` / ``--reasoning-parser`` — fanned out to both.
   ``--reasoning-parser`` goes to the gateway (post-gen parsing) and
   the engine (defers JSON grammars past the reasoning channel).
3. ``--host`` / ``--port`` — gateway only (user-facing)
4. ``--chat-template`` / ``--tool-call-parser`` — gateway only
5. ``--tp`` / ``--tensor-parallel-size`` — engine only (alias normalized)
6. Anything else ``prepare_server_args`` accepts — engine only
7. Anything else — gateway (fall-through to ``smg launch`` clap)
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass, field
from typing import Iterable

_ORCH_FLAGS = {
    "--engine-startup-timeout",
    "--gateway-startup-timeout",
    "--drain-timeout",
}

_FANOUT_FLAGS = {"--model", "--reasoning-parser"}

_ALIASES = {
    "--model-path": "--model",
    "--tp": "--tensor-parallel-size",
}

_GATEWAY_USER_FACING = {"--host", "--port"}

_GATEWAY_OVERRIDE = {
    "--chat-template",
    "--tool-call-parser",
}

_ENGINE_EXPLICIT = {"--tensor-parallel-size"}

_MODEL_FLAG_TOKENS = ("--model", "--model-path")

_HEADLESS_FLAG = "--headless"

_ENGINE_MULTI_VALUE_FLAGS = {
    "--cudagraph-capture-sizes",
    "--prefill-graph-capture-sizes",
    "--prefill-graph-capture-token-sizes",
    "--prefill-graph-capture-batch-sizes",
}


def _has_model_flag(tokens: Iterable[str]) -> bool:
    for token in tokens:
        if token in _MODEL_FLAG_TOKENS:
            return True
        for flag in _MODEL_FLAG_TOKENS:
            if token.startswith(flag + "="):
                return True
    return False


@dataclass
class OrchestratorOpts:
    engine_startup_timeout: int = 3600
    gateway_startup_timeout: int = 60
    drain_timeout: int = 30


@dataclass
class SplitResult:
    engine: list[str] = field(default_factory=list)
    gateway: list[str] = field(default_factory=list)
    opts: OrchestratorOpts = field(default_factory=OrchestratorOpts)


def _normalize(
    argv: Iterable[str], engine_flags: set[str]
) -> list[tuple[str, list[str] | str | None]]:
    """Convert raw argv into a list of (name, value) pairs.

    Handles both ``--flag value`` and ``--flag=value`` forms. Aliases are
    resolved to their canonical names. Gateway-only options keep all consecutive
    values; SMG validates their arity. Known scalar options still take one value.
    """
    items: list[tuple[str, list[str] | str | None]] = []
    tokens = list(argv)
    scalar_flags = (
        engine_flags
        | _ORCH_FLAGS
        | _FANOUT_FLAGS
        | _GATEWAY_USER_FACING
        | _GATEWAY_OVERRIDE
        | _ENGINE_EXPLICIT
    )
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        if not raw.startswith("--"):
            raise ValueError(f"unexpected positional arg: {raw!r}")
        if "=" in raw:
            name, _, value = raw.partition("=")
            name = _ALIASES.get(name, name)
            i += 1
        else:
            name = _ALIASES.get(raw, raw)
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or nxt.startswith("--"):
                value = None
                i += 1
            elif name in _ENGINE_MULTI_VALUE_FLAGS or name not in scalar_flags:
                values = []
                i += 1
                while i < len(tokens) and not tokens[i].startswith("--"):
                    values.append(tokens[i])
                    i += 1
                value = values
            else:
                value = nxt
                i += 2
        items.append((name, value))
    return items


def _append_arg(args: list[str], name: str, value: list[str] | str | None) -> None:
    if value is None:
        args.append(name)
    elif isinstance(value, list):
        args.extend([name, *value])
    else:
        args.extend([name, value])


def split_headless_argv(argv: list[str]) -> list[str] | None:
    """Extract the ServerArgs argv for ``ts serve --headless``.

    Returns ``None`` when ``--headless`` is absent (default orchestrator
    mode). Otherwise returns the argv with the flag removed, for
    ``prepare_server_args`` verbatim: the engine is the only process, so
    positional-model and alias handling belong to ServerArgs' own parser
    and no engine/gateway routing applies.

    Raises:
        ValueError: if an orchestrator-only flag is present. Headless mode
            has no gateway lifecycle to configure, so rejecting loudly beats
            silently ignoring the flag.
    """
    if _HEADLESS_FLAG not in argv:
        return None
    tokens = [token for token in argv if token != _HEADLESS_FLAG]
    for token in tokens:
        name = token.partition("=")[0]
        if name in _ORCH_FLAGS:
            raise ValueError(
                f"{name} configures the gateway orchestrator and is not "
                "valid with --headless (no gateway is spawned)"
            )
    return tokens


# Architecture assumed for the parser-only platform when no device is visible.
_PARSER_PLATFORM_ARCH_ENV = "TOKENSPEED_ARGSPLIT_PARSER_ARCH"
_parser_platform: str | None = None


def _ensure_parser_platform() -> str:
    """Make ``ServerArgs`` importable on a host with no visible device.

    Importing ``tokenspeed.runtime.utils.server_args`` imports the kernel
    package, and modules there call ``tokenspeed_kernel.platform.current_platform()``
    at import time (e.g. ``ops/quantization/flashinfer.py``), which raises
    "tokenspeed-kernel requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device."
    when none is visible. Splitting argv only needs the parser's flag names, so
    when detection fails this installs a placeholder platform through
    ``Platform.override`` (device_count 0; arch from
    ``TOKENSPEED_ARGSPLIT_PARSER_ARCH``, default gfx950). With a visible device
    the real platform is used and nothing is overridden.

    Returns "detected:<vendor>-<arch>" or "placeholder:<arch>".
    """
    global _parser_platform
    if _parser_platform is not None:
        return _parser_platform
    from tokenspeed_kernel.platform import Platform, current_platform

    try:
        info = current_platform()
        _parser_platform = f"detected:{info.vendor}-{info.arch_version}"
        return _parser_platform
    except RuntimeError as error:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" not in str(error):
            raise
    import logging
    import os

    from tokenspeed_kernel.platform import (
        ArchVersion,
        PlatformInfo,
        _get_rocm_sm_features,
    )

    arch = os.environ.get(_PARSER_PLATFORM_ARCH_ENV, "gfx950")
    versions = {"gfx942": ArchVersion(9, 4), "gfx950": ArchVersion(9, 5)}
    if arch not in versions:
        raise ValueError(
            f"{_PARSER_PLATFORM_ARCH_ENV}={arch!r}: expected one of {sorted(versions)}"
        )
    Platform.override(
        PlatformInfo(
            vendor="amd",
            arch_version=versions[arch],
            device_name=f"argsplit parser placeholder ({arch}, no visible device)",
            device_count=0,
            total_memory=0,
            memory_bandwidth=0.0,
            sm_count=0,
            max_threads_per_sm=0,
            max_shared_memory_per_sm=0,
            sm_features=_get_rocm_sm_features(arch),
        )
    )
    logging.getLogger(__name__).info(
        "no visible device: ServerArgs flags read with a placeholder %s platform", arch
    )
    _parser_platform = f"placeholder:{arch}"
    return _parser_platform


@functools.lru_cache(maxsize=1)
def _engine_recognized_flags() -> set[str]:
    """Snapshot the set of long-form flags accepted by ``prepare_server_args``."""
    _ensure_parser_platform()
    # Lazy import: ServerArgs pulls the full runtime stack (~200ms).
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    flags: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    flags.discard("--help")
    flags.discard("-h")
    return flags


def split_argv(argv: list[str]) -> SplitResult:
    """Split ts-serve argv into engine_args, gateway_args, orchestrator_opts.

    A leading positional argument is rewritten to ``--model <value>`` so
    ``ts serve <model> [flags...]`` and ``ts serve --model <model> [flags...]``
    both work.

    Raises:
        ValueError: if a flag that requires a value is provided without one
            (e.g. ``--model`` with no path), if a timeout flag is
            non-positive, if the model is given both positionally and via
            ``--model``/``--model-path``, or if a positional arg appears
            after the leading model without belonging to a gateway option.
    """

    argv = list(argv)
    if argv and not argv[0].startswith("--"):
        model = argv[0]
        rest = argv[1:]
        if _has_model_flag(rest):
            raise ValueError(
                "model specified both as positional argument and via "
                "--model/--model-path"
            )
        argv = ["--model", model, *rest]

    engine_flags = _engine_recognized_flags()
    items = _normalize(argv, engine_flags)
    result = SplitResult()

    for name, value in items:
        if name in _ORCH_FLAGS:
            if value is None or value == "":
                raise ValueError(f"{name} requires a positive integer (seconds)")
            try:
                seconds = int(value)
            except ValueError as e:
                raise ValueError(f"{name}={value!r} is not a valid integer") from e
            if seconds <= 0:
                raise ValueError(f"{name} must be positive, got {seconds}")
            attr = name[2:].replace("-", "_")
            setattr(result.opts, attr, seconds)
            continue

        if name in _FANOUT_FLAGS:
            if value is None:
                raise ValueError(f"{name} requires a value")
            _append_arg(result.engine, name, value)
            _append_arg(result.gateway, name, value)
            continue

        if name in _GATEWAY_USER_FACING:
            if value is None:
                raise ValueError(f"{name} requires a value")
            _append_arg(result.gateway, name, value)
            continue

        if name in _GATEWAY_OVERRIDE:
            if value is None:
                raise ValueError(f"{name} requires a value")
            _append_arg(result.gateway, name, value)
            continue

        if name in _ENGINE_EXPLICIT:
            if value is None:
                raise ValueError(f"{name} requires a value")
            _append_arg(result.engine, name, value)
            continue

        if name in engine_flags:
            _append_arg(result.engine, name, value)
            continue

        _append_arg(result.gateway, name, value)

    return result


_GATEWAY_PARSER_WORKER_URL = "http://127.0.0.1:30000"


def _parse_quietly(parse, args: list[str]) -> tuple[bool, str]:
    """Run an argparse-based parser; return (accepted, captured error text)."""
    import contextlib
    import io

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parse(args)
        return True, ""
    except SystemExit as error:
        return False, err.getvalue().strip() or f"SystemExit {error.code}"


def _unrecognized(error: str) -> list[str]:
    marker = "unrecognized arguments:"
    if marker not in error:
        return []
    tail = error.rsplit(marker, 1)[1].strip().splitlines()[0]
    return [token for token in tail.split() if token.startswith("--")]


def check_argv(argv: list[str]) -> dict:
    """Route ``ts serve`` argv and check each side with its real parser.

    ``argv`` is what follows the ``ts`` executable; a leading ``serve`` is
    dropped. The engine side is parsed as ``prepare_server_args`` parses it
    (``argparse.ArgumentParser(allow_abbrev=False)`` + ``ServerArgs.add_cli_args``,
    server_args.py ``prepare_server_args``) without building ``ServerArgs``
    (that resolves the model and the hardware). The gateway side is parsed with
    ``smg.launch_router.parse_router_args`` after ``--worker-urls``, as this
    module's tests do. ``unknown`` lists flags neither parser defines.

    Returns a JSON-serializable dict; ``ok`` is True when both parsers accept
    their side and no flag is unknown.
    """
    argv = list(argv)
    if argv[:1] == ["serve"]:
        argv = argv[1:]
    platform = _ensure_parser_platform()
    split = split_argv(argv)
    engine_flags = _engine_recognized_flags()
    from tokenspeed.runtime.utils.server_args import ServerArgs

    def engine_parse(args: list[str]) -> None:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        ServerArgs.add_cli_args(parser)
        parser.parse_args(args)

    engine_ok, engine_error = _parse_quietly(engine_parse, split.engine)
    try:
        from smg.launch_router import parse_router_args
    except ImportError as error:  # tokenspeed-smg missing
        gateway_ok, gateway_error = False, f"cannot import smg.launch_router: {error}"
    else:
        gateway_ok, gateway_error = _parse_quietly(
            parse_router_args,
            ["--worker-urls", _GATEWAY_PARSER_WORKER_URL, *split.gateway],
        )

    def names(tokens: list[str]) -> list[str]:
        return [t.partition("=")[0] for t in tokens if t.startswith("--")]

    engine_names, gateway_names = names(split.engine), names(split.gateway)
    routing = []
    for name, _ in _normalize(argv, engine_flags):
        if name in _ORCH_FLAGS:
            routed = "orchestrator"
        else:
            routed = "+".join(
                side
                for side, seen in (("engine", engine_names), ("gateway", gateway_names))
                if name in seen
            )
        routing.append({"flag": name, "routed_to": routed})
    unknown = sorted(set(_unrecognized(gateway_error)) - engine_flags)
    flags_digest = (
        __import__("hashlib")
        .sha256("\n".join(sorted(engine_flags)).encode())
        .hexdigest()
    )
    return {
        "schema_version": 1,
        "parser_platform": platform,
        "engine_flags_count": len(engine_flags),
        "engine_flags_sha256": flags_digest,
        "engine_argv": split.engine,
        "gateway_argv": split.gateway,
        "orchestrator": vars(split.opts),
        "both": [n for n in engine_names if n in gateway_names],
        "unknown": unknown,
        "routing": routing,
        "engine_parser_accepts": engine_ok,
        "engine_parser_error": engine_error,
        "gateway_parser_accepts": gateway_ok,
        "gateway_parser_error": gateway_error,
        "ok": engine_ok and gateway_ok and not unknown,
    }


def check_server_args(engine_argv: list[str], gateway_argv: list[str]) -> dict:
    """Resolve the engine's ``ServerArgs`` as ``ts serve`` does, without launching.

    ``ts serve`` (cli/serve_smg.py ``main``) applies
    ``_args_with_default_model_parsers`` to the split argv and then calls
    ``_engine_layout(engine_args)``, i.e. ``prepare_server_args`` ->
    ``ServerArgs.from_cli_args`` (all validation that runs before any engine
    process starts) plus its disaggregation check. This runs those same two
    calls. It needs the real platform (a visible device) and reads the model
    config; it starts no process. Returns ``server_args_ok`` and, on failure,
    ``server_args_error`` as ``"<ExceptionType>: <message>"`` (verbatim) and
    the last traceback frame.
    """
    import traceback

    from tokenspeed.cli import serve_smg

    engine_args, _ = serve_smg._args_with_default_model_parsers(
        list(engine_argv), list(gateway_argv)
    )
    try:
        layout = serve_smg._engine_layout(engine_args)
    except BaseException as error:  # noqa: BLE001 - record any refusal verbatim
        frame = traceback.extract_tb(error.__traceback__)[-1]
        return {
            "server_args_ok": False,
            "server_args_error": f"{type(error).__name__}: {error}",
            "server_args_error_at": f"{frame.filename}:{frame.lineno}",
        }
    return {
        "server_args_ok": True,
        "server_args_error": "",
        "engine_layout": {
            "count": layout.count,
            "remote_handshake": layout.remote_handshake,
        },
    }


def check_moe_plans(engine_argv: list[str]) -> dict:
    """Select the target and draft MoE apply kernels of DeepSeek V4.1 + DSpark.

    Runs TokenSpeed's own kernel selection, ``tokenspeed_kernel.ops.moe.moe_plan``
    (the call ``MoELayer.__init__`` makes, layers/moe/expert.py:512-548), with the
    arguments that layer derives from the server flags and the model config, so it
    needs no device and no weights (the placeholder platform stands in when no
    device is visible). Inputs, each from the e8fd096e code path it mirrors:

    - backend: ``--moe-backend`` for the target; ``--draft-moe-backend`` (else
      ``--moe-backend``) for the DSpark draft (execution/model_runner.py:184-189);
      ``auto`` -> no forced solution, ``gluon_petit`` -> ``gluon``,
      ``deep_gemm_mega_moe`` -> ``mega_moe`` (layers/moe/expert.py:368-374);
    - a2a_backend: ``--all2all-backend``, ``agrs``/``flashinfer`` -> ``none``
      (expert.py:166-168);
    - geometry: ``text_config`` n_routed_experts / num_experts_per_tok for the
      target, dspark_n_routed_experts / dspark_num_experts_per_tok for the draft
      (models/deepseek_v41_dspark.py:323-324); hidden_size; ispp =
      moe_intermediate_size // --moe-tp-size, padded only for mega_moe
      (models/deepseek_v41.py:1822-1840);
    - activation swiglu with ``swiglu_limit``, precomputed top-k, no bias, no
      fp32 correction bias (models/deepseek_v41_moe.py:310-336); swiglu_form from
      ``MoELayer._swiglu_form`` (expert.py:585-594);
    - weight dtype and activation dtype from the routed-expert quant config
      ``_deepseek_v4_routed_expert_quant_config`` (deepseek_v41_moe.py:174-192),
      ``--moe-mxfp4-fp8-activation`` and Petit (expert.py:226-266, 271-339).

    Returns ``{"moe_plan": {"target": {...}, "draft": {...}}}``; an entry whose
    selection raises carries ``error`` instead of ``apply_kernel_name``.
    """
    import json
    import os
    from types import SimpleNamespace

    import torch

    _ensure_parser_platform()
    from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan

    from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
    from tokenspeed.runtime.models.deepseek_v41_moe import (
        _deepseek_v4_routed_expert_quant_config,
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser(allow_abbrev=False)
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(engine_argv)
    # Parser dests: --ep-size is an alias of --expert-parallel-size.
    args.ep_size = args.expert_parallel_size
    if args.moe_tp_size is None:
        raise ValueError("--moe-tp-size must be explicit for --moe-plan")
    with open(os.path.join(args.model, "config.json")) as handle:
        hf = json.load(handle)
    text = SimpleNamespace(**hf.get("text_config", hf))
    quant = hf.get("quantization_config") or {}
    text.expert_dtype = getattr(text, "expert_dtype", quant.get("expert_dtype"))
    a2a = args.all2all_backend
    a2a = "none" if a2a in ("agrs", "flashinfer") else a2a
    roles = {
        "target": (args.moe_backend, text.n_routed_experts, text.num_experts_per_tok),
        "draft": (
            args.draft_moe_backend or args.moe_backend,
            getattr(text, "dspark_n_routed_experts", None),
            getattr(text, "dspark_num_experts_per_tok", None),
        ),
    }
    bitwise = getattr(args, "numerics", "auto") in BITWISE_ENVELOPES
    out = {}
    for role, (backend, num_experts, top_k) in roles.items():
        entry = {
            "moe_backend": backend,
            "a2a_backend": a2a,
            "ep_size": args.ep_size,
            "moe_tp_size": args.moe_tp_size,
            "num_experts": num_experts,
            "top_k": top_k,
        }
        out[role] = entry
        if num_experts is None:
            entry["error"] = "model config has no DSpark draft experts"
            continue
        solution = {"deep_gemm_mega_moe": "mega_moe", "gluon_petit": "gluon"}.get(
            backend, backend
        )
        solution = None if solution == "auto" else solution
        ispp_full = text.moe_intermediate_size
        if solution == "mega_moe" and ispp_full % 512:
            # deepseek_v41.py:1822-1840 (non-Blackwell alignment 512)
            ispp_full = (ispp_full + 511) // 512 * 512
        routed_quant, _ = _deepseek_v4_routed_expert_quant_config(
            text, SimpleNamespace(ignored_layers=[])
        )
        prefix = "model.layers.0.ffn"
        layer_quant = routed_quant.get_moe_quant_config(prefix)
        quant_kind = layer_quant.moe_weight_dtype(prefix)
        act_dtype = "input"
        if quant_kind == "mxfp4":
            if layer_quant.is_w4a8_fp8:
                act_dtype = "fp8"
            elif getattr(layer_quant, "use_dynamic_mxfp4_activations", False):
                act_dtype = "mxfp4"
        if args.moe_mxfp4_fp8_activation:
            act_dtype = "fp8"
        if a2a == "gluon_petit":
            act_dtype = "mxfp4"
        swiglu_limit = getattr(text, "swiglu_limit", None)
        entry.update(
            weight_dtype=quant_kind,
            ispp=ispp_full // args.moe_tp_size,
            hidden=text.hidden_size,
            internal_activation_dtype=act_dtype,
        )
        try:
            plan = kernel_moe_plan(
                quant_kind,
                input_dtype=torch.bfloat16,
                activation="swiglu",
                routing_mode="precomputed_topk",
                a2a_backend=a2a,
                ep_size=args.ep_size,
                ispp=ispp_full // args.moe_tp_size,
                hidden=text.hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                swiglu_form="standard",
                activation_clamped=swiglu_limit is not None,
                expert_id_repeats=False,
                fp8_scale_block_shape=None,
                internal_activation_dtype=act_dtype,
                with_bias=False,
                process_group=None,
                persistent_max_num_tokens_per_gpu=(
                    1 if solution == "mega_moe" else None
                ),
                features=frozenset({"batch_invariant"}) if bitwise else frozenset(),
                solution=solution,
                fast_math=not bitwise,
                combine_order=args.moe_combine_order,
                fp32_correction_bias=False,
            )
        except Exception as error:  # noqa: BLE001 - report the refusal verbatim
            entry["error"] = f"{type(error).__name__}: {error}"
            continue
        entry["apply_kernel_name"] = plan["apply_kernel_name"]
        entry["solution"] = plan["solution"]
    return {"moe_plan": out}


def main(argv: list[str] | None = None) -> int:
    """``python -m tokenspeed.cli._argsplit --check (--argv-json FILE | -- ARGV...)``.

    ``--argv-json`` reads a JSON list of argv tokens, or an object with an
    ``argv`` list (a recipe's ``--print-argv`` output); ``-`` reads stdin.
    Prints the ``check_argv`` result as one JSON object; exit 0 when ``ok``.
    """
    import json
    import sys

    parser = argparse.ArgumentParser(prog="python -m tokenspeed.cli._argsplit")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--argv-json")
    parser.add_argument("--label", help="copied into the output as 'label'")
    parser.add_argument(
        "--server-args",
        action="store_true",
        help="also resolve ServerArgs as `ts serve` does before launching "
        "(needs a visible device and the model config); exit 1 if it fails",
    )
    parser.add_argument(
        "--moe-plan",
        action="store_true",
        help="also report the target and draft MoE apply kernel TokenSpeed's "
        "moe_plan selects for DeepSeek V4.1 + DSpark (reads the model's "
        "config.json; needs no device); exit 1 if either selection fails",
    )
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    tokens = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if args.argv_json is not None:
        if tokens:
            parser.error("give --argv-json or argv after --, not both")
        source = sys.stdin if args.argv_json == "-" else open(args.argv_json)
        with source:
            loaded = json.load(source)
        tokens = loaded["argv"] if isinstance(loaded, dict) else loaded
    result = check_argv(tokens)
    if args.server_args:
        result.update(check_server_args(result["engine_argv"], result["gateway_argv"]))
        result["ok"] = result["ok"] and result["server_args_ok"]
    if args.moe_plan:
        result.update(check_moe_plans(result["engine_argv"]))
        result["ok"] = result["ok"] and not any(
            "error" in entry for entry in result["moe_plan"].values()
        )
    if args.label is not None:
        result = {"label": args.label, **result}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

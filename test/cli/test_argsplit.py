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

from __future__ import annotations

import pytest

from tokenspeed.cli._argsplit import SplitResult, split_argv


def _split(argv: list[str]) -> SplitResult:
    return split_argv(argv)


def test_orchestrator_only_flags_are_consumed():
    r = _split(["--engine-startup-timeout", "300"])
    assert r.engine == []
    assert r.gateway == []
    assert r.opts.engine_startup_timeout == 300


def test_orchestrator_default_timeouts():
    r = _split([])
    assert r.opts.engine_startup_timeout == 3600
    assert r.opts.gateway_startup_timeout == 60
    assert r.opts.drain_timeout == 30


def test_model_fans_out_to_both():
    r = _split(["--model", "/models/qwen3-4b"])
    assert "--model" in r.engine and "/models/qwen3-4b" in r.engine
    assert "--model" in r.gateway and "/models/qwen3-4b" in r.gateway


def test_model_path_alias_also_fans_out():
    """``--model-path`` is accepted as an alias and normalizes to ``--model``."""
    r = _split(["--model-path", "/models/qwen3-4b"])
    assert "--model" in r.engine and "/models/qwen3-4b" in r.engine
    assert "--model" in r.gateway and "/models/qwen3-4b" in r.gateway
    # The alias name must not leak through.
    assert "--model-path" not in r.engine
    assert "--model-path" not in r.gateway


def test_user_host_port_go_to_gateway_only():
    r = _split(["--host", "0.0.0.0", "--port", "8000"])
    assert r.engine == []  # engine binds 127.0.0.1:<auto> internally
    assert r.gateway == ["--host", "0.0.0.0", "--port", "8000"]


def test_chat_template_overrides_to_gateway_only():
    r = _split(["--chat-template", "/some/template.jinja"])
    assert r.engine == []
    assert r.gateway == ["--chat-template", "/some/template.jinja"]


def test_api_key_routes_to_gateway():
    r = _split(["--model", "test/model", "--api-key", "test-key"])
    assert r.engine == ["--model", "test/model"]
    assert r.gateway == ["--model", "test/model", "--api-key", "test-key"]


def test_tool_call_parser_overrides_to_gateway_only():
    r = _split(["--tool-call-parser", "hermes"])
    assert r.engine == []
    assert r.gateway == ["--tool-call-parser", "hermes"]


def test_reasoning_parser_fans_out_to_both():
    # Gateway uses it for post-gen reasoning parsing; engine uses it to
    # defer JSON grammars past the reasoning channel.
    r = _split(["--reasoning-parser", "qwen3"])
    assert r.engine == ["--reasoning-parser", "qwen3"]
    assert r.gateway == ["--reasoning-parser", "qwen3"]


def test_tp_alias_routes_to_engine():
    r = _split(["--tp", "2"])
    assert r.engine == ["--tensor-parallel-size", "2"]
    assert r.gateway == []


def test_tensor_parallel_size_routes_to_engine():
    r = _split(["--tensor-parallel-size", "4"])
    assert r.engine == ["--tensor-parallel-size", "4"]
    assert r.gateway == []


def test_engine_only_flag_routes_to_engine():
    """Anything prepare_server_args accepts but smg does not."""
    r = _split(["--sampling-backend", "flashinfer"])
    assert r.engine == ["--sampling-backend", "flashinfer"]
    assert r.gateway == []


def test_dp_sampling_routes_to_engine():
    r = _split(["--dp-sampling"])
    assert r.engine == ["--dp-sampling"]
    assert r.gateway == []


def test_kda_prefill_graph_routes_to_engine():
    r = _split(["--disable-kda-prefill-graph"])
    assert r.engine == ["--disable-kda-prefill-graph"]
    assert r.gateway == []


def test_multi_value_capture_sizes_route_to_engine():
    """nargs='+' engine flags: every bare value belongs to the flag, not argv."""
    for flag in (
        "--cudagraph-capture-sizes",
        "--prefill-graph-capture-sizes",
        "--prefill-graph-capture-token-sizes",
        "--prefill-graph-capture-batch-sizes",
    ):
        r = _split([flag, "288", "320", "--dp-sampling"])
        assert r.engine == [flag, "288", "320", "--dp-sampling"]
        assert r.gateway == []


def test_unknown_flag_falls_through_to_gateway():
    """smg's clap is the final authority on these."""
    r = _split(["--policy", "cache_aware"])
    assert r.engine == []
    assert r.gateway == ["--policy", "cache_aware"]


def test_combined_real_invocation():
    argv = [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--port",
        "8000",
        "--tp",
        "2",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--sampling-backend",
        "flashinfer",
        "--policy",
        "cache_aware",
    ]
    r = _split(argv)
    # Engine: --model (fan-out), --tp (normalized), --reasoning-parser
    # (fan-out), --sampling-backend. Argv order is preserved.
    assert r.engine == [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--tensor-parallel-size",
        "2",
        "--reasoning-parser",
        "qwen3",
        "--sampling-backend",
        "flashinfer",
    ]
    # Gateway: --model (fan-out), --port, parsers, --policy.
    assert r.gateway == [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--port",
        "8000",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--policy",
        "cache_aware",
    ]


def test_equals_form_is_normalized():
    r = _split(["--model=/models/qwen3", "--port=8000"])
    assert r.engine == ["--model", "/models/qwen3"]
    assert r.gateway == ["--model", "/models/qwen3", "--port", "8000"]


def test_valueless_flag_followed_by_equals_form():
    """Regression: --foo --bar=x must not glue =x to --bar's name."""
    r = _split(["--enable-foo", "--port=8000"])
    # --enable-foo is unknown to engine introspection, so falls through.
    assert r.gateway == ["--enable-foo", "--port", "8000"]


def test_valueless_flag_followed_by_value_flag():
    """Regression: --foo --bar value must not lose 'value'."""
    r = _split(["--enable-foo", "--port", "8000"])
    assert r.gateway == ["--enable-foo", "--port", "8000"]


def test_timeout_with_non_integer_raises():
    with pytest.raises(ValueError, match="not a valid integer"):
        _split(["--engine-startup-timeout", "abc"])


def test_timeout_with_empty_value_raises():
    with pytest.raises(ValueError, match="requires a positive integer"):
        _split(["--engine-startup-timeout="])


def test_timeout_with_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        _split(["--engine-startup-timeout", "0"])


def test_leading_positional_is_treated_as_model():
    r = _split(
        [
            "openai/gpt-oss-20b",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tensor-parallel-size",
            "1",
        ]
    )
    assert "--model" in r.engine and "openai/gpt-oss-20b" in r.engine
    assert "--model" in r.gateway and "openai/gpt-oss-20b" in r.gateway
    assert r.gateway[r.gateway.index("--host") + 1] == "0.0.0.0"
    assert r.gateway[r.gateway.index("--port") + 1] == "8000"
    assert r.engine[r.engine.index("--tensor-parallel-size") + 1] == "1"


def test_positional_model_matches_explicit_model_flag():
    """Both invocation styles must produce the same split."""
    positional = _split(["openai/gpt-oss-20b", "--host", "0.0.0.0"])
    explicit = _split(["--model", "openai/gpt-oss-20b", "--host", "0.0.0.0"])
    assert positional == explicit


def test_positional_and_model_flag_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model", "other/model"])


def test_positional_and_model_path_alias_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model-path", "other/model"])


def test_positional_and_model_equals_form_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model=other/model"])


def test_trailing_positional_still_raises():
    """Only the FIRST argument may be positional; later ones remain illegal."""
    with pytest.raises(ValueError, match="unexpected positional arg"):
        _split(["--host", "0.0.0.0", "stray-positional"])


@pytest.mark.parametrize(
    "flag,values",
    [
        (
            "--cors-allowed-origins",
            ["https://app.example.com", "https://admin.example.com"],
        ),
        ("--routing-key-headers", ["x-session-id", "x-user-id"]),
        ("--access-log-fields", ["client_addr", "user_agent"]),
    ],
)
def test_gateway_multi_value_flags_preserve_all_values(flag, values):
    result = _split(["example/model", flag, *values, "--tp", "2", "--port", "8000"])
    assert result.engine == ["--model", "example/model", "--tensor-parallel-size", "2"]
    assert result.gateway == [
        "--model",
        "example/model",
        flag,
        *values,
        "--port",
        "8000",
    ]


def test_gateway_multi_value_flag_can_be_empty():
    result = _split(["--cors-allowed-origins", "--port", "8000"])
    assert result.gateway == ["--cors-allowed-origins", "--port", "8000"]
    assert result.engine == []


def test_engine_scalar_flag_still_rejects_extra_values():
    with pytest.raises(ValueError, match="unexpected positional arg"):
        _split(["--max-model-len", "4096", "8192"])


@pytest.mark.parametrize("equals_form", [False, True])
def test_gateway_cors_arguments_reach_smg_parser(equals_form):
    pytest.importorskip("smg")
    from smg.launch_router import parse_router_args

    origins = ["https://app.example.com", "https://admin.example.com"]
    cors_args = (
        [f"--cors-allowed-origins={origins[0]}"]
        if equals_form
        else ["--cors-allowed-origins", *origins]
    )
    result = _split([*cors_args, "--port", "8100"])
    parsed = parse_router_args(
        ["--worker-urls", "http://127.0.0.1:30000", *result.gateway]
    )
    assert parsed.cors_allowed_origins == (origins[:1] if equals_form else origins)
    assert parsed.port == 8100


def test_gateway_parser_rejects_extra_values_for_scalar_flag():
    pytest.importorskip("smg")
    from smg.launch_router import parse_router_args

    result = _split(["--policy", "passthrough", "extra"])
    with pytest.raises(SystemExit) as error:
        parse_router_args(["--worker-urls", "http://127.0.0.1:30000", *result.gateway])
    assert error.value.code == 2


# --- check_argv / ``--check``: routing plus both real parsers ---------------------

_RECIPE_LIKE = [
    "serve",
    "--model",
    "/models/m",
    "--attn-tp-size",
    "4",
    "--kv-cache-dtype",
    "fp8",
    "--cudagraph-capture-sizes",
    "1",
    "2",
    "4",
    "--reasoning-parser",
    "passthrough",
    "--tool-call-parser",
    "passthrough",
    "--host",
    "127.0.0.1",
    "--port",
    "30000",
    "--policy",
    "round_robin",
    "--engine-startup-timeout",
    "7200",
]


def test_check_argv_routes_and_accepts_recipe_like_argv():
    pytest.importorskip("smg")
    from tokenspeed.cli._argsplit import check_argv

    r = check_argv(_RECIPE_LIKE)
    assert r["ok"], r
    assert r["both"] == ["--model", "--reasoning-parser"]
    routed = {row["flag"]: row["routed_to"] for row in r["routing"]}
    assert routed["--attn-tp-size"] == "engine"
    assert routed["--policy"] == "gateway"
    assert routed["--engine-startup-timeout"] == "orchestrator"
    assert r["orchestrator"]["engine_startup_timeout"] == 7200


def test_check_argv_reports_flags_no_parser_knows():
    pytest.importorskip("smg")
    from tokenspeed.cli._argsplit import check_argv

    r = check_argv([*_RECIPE_LIKE, "--no-enable-prefix-caching", "--control-port", "1"])
    assert not r["ok"]
    assert r["unknown"] == ["--control-port", "--no-enable-prefix-caching"]
    assert "unrecognized arguments" in r["gateway_parser_error"]


def test_check_argv_reports_engine_parser_rejection():
    pytest.importorskip("smg")
    from tokenspeed.cli._argsplit import check_argv

    argv = list(_RECIPE_LIKE)
    argv[argv.index("fp8")] = "fp16"
    r = check_argv(argv)
    assert not r["engine_parser_accepts"] and not r["ok"]
    assert "invalid choice: 'fp16'" in r["engine_parser_error"]


def test_check_cli_runs_without_a_visible_device(tmp_path):
    """``--check`` (and split_argv) import ServerArgs with no GPU visible."""
    pytest.importorskip("smg")
    import inspect
    import json
    import os
    import subprocess
    import sys

    from tokenspeed.cli._argsplit import check_argv

    recipe = tmp_path / "argv.json"
    recipe.write_text(json.dumps({"layout": "t", "argv": _RECIPE_LIKE}))
    env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    env.pop("ROCR_VISIBLE_DEVICES", None)
    out = subprocess.run(
        [
            sys.executable,
            inspect.getfile(check_argv),
            "--check",
            "--argv-json",
            str(recipe),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout)
    assert r["ok"] and r["parser_platform"].startswith(("placeholder:", "detected:"))


def _fake_serve_smg(monkeypatch, layout_fn):
    import sys
    import types

    import tokenspeed.cli as cli_pkg

    fake = types.ModuleType("tokenspeed.cli.serve_smg")
    fake._args_with_default_model_parsers = lambda e, g: (list(e) + ["--x-default"], g)
    fake._engine_layout = layout_fn
    monkeypatch.setitem(sys.modules, "tokenspeed.cli.serve_smg", fake)
    monkeypatch.setattr(cli_pkg, "serve_smg", fake, raising=False)


def test_check_server_args_reports_layout(monkeypatch):
    import types

    from tokenspeed.cli._argsplit import check_server_args

    seen = []

    def layout(engine_args):
        seen.append(engine_args)
        return types.SimpleNamespace(count=4, remote_handshake=None)

    _fake_serve_smg(monkeypatch, layout)
    r = check_server_args(["--model", "m"], ["--model", "m"])
    assert r == {
        "server_args_ok": True,
        "server_args_error": "",
        "engine_layout": {"count": 4, "remote_handshake": None},
    }
    assert seen == [["--model", "m", "--x-default"]]


def test_check_server_args_records_refusal_verbatim(monkeypatch):
    from tokenspeed.cli._argsplit import check_server_args

    def layout(engine_args):
        raise ValueError("chunked_prefill_size must be <= max_prefill_tokens")

    _fake_serve_smg(monkeypatch, layout)
    r = check_server_args(["--model", "m"], [])
    assert not r["server_args_ok"]
    assert r["server_args_error"] == (
        "ValueError: chunked_prefill_size must be <= max_prefill_tokens"
    )
    assert r["server_args_error_at"].endswith(
        ".py:" + r["server_args_error_at"].rsplit(":", 1)[1]
    )


# --- check_moe_plans / ``--moe-plan``: TokenSpeed's MoE kernel selection ------------

_V41_TEXT_CONFIG = {
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "n_routed_experts": 384,
    "num_experts_per_tok": 6,
    "swiglu_limit": 10.0,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
}


def _v41_model_dir(tmp_path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
                "text_config": _V41_TEXT_CONFIG,
            }
        )
    )
    return str(tmp_path)


def _layout_argv(model: str, moe_tp: int, ep: int, *extra: str) -> list[str]:
    return [
        "--model",
        model,
        "--moe-tp-size",
        str(moe_tp),
        "--ep-size",
        str(ep),
        *extra,
    ]


def test_check_moe_plans_passes_layer_inputs_to_moe_plan(tmp_path, monkeypatch):
    from tokenspeed.cli._argsplit import _ensure_parser_platform, check_moe_plans

    _ensure_parser_platform()  # the kernel package needs a platform at import
    import tokenspeed_kernel.ops.moe as kernel_moe

    calls = []

    def fake_plan(weight_dtype, **kwargs):
        calls.append((weight_dtype, kwargs))
        return {"apply_kernel_name": f"k{len(calls)}", "solution": "s"}

    monkeypatch.setattr(kernel_moe, "moe_plan", fake_plan)
    model = _v41_model_dir(tmp_path)
    r = check_moe_plans(
        _layout_argv(
            model,
            1,
            4,
            "--moe-backend",
            "gluon_petit",
            "--draft-moe-backend",
            "gluon_petit",
            "--all2all-backend",
            "gluon_petit",
        )
    )["moe_plan"]
    assert r["target"]["apply_kernel_name"] == "k1"
    assert r["draft"]["apply_kernel_name"] == "k2"
    (wt, target), (wd, draft) = calls
    assert wt == wd == "mxfp4"
    assert (target["num_experts"], target["top_k"]) == (384, 6)
    assert (draft["num_experts"], draft["top_k"]) == (128, 3)
    for kw in (target, draft):
        assert kw["solution"] == "gluon" and kw["a2a_backend"] == "gluon_petit"
        assert kw["internal_activation_dtype"] == "mxfp4"
        assert kw["ep_size"] == 4 and kw["ispp"] == 2304 and kw["hidden"] == 5120
        assert kw["activation"] == "swiglu" and kw["routing_mode"] == "precomputed_topk"
        assert kw["activation_clamped"] is True and kw["swiglu_form"] == "standard"

    calls.clear()
    r = check_moe_plans(_layout_argv(model, 4, 1, "--moe-backend", "auto"))["moe_plan"]
    assert [kw["solution"] for _, kw in calls] == [None, None]
    assert [kw["ispp"] for _, kw in calls] == [576, 576]
    assert [kw["a2a_backend"] for _, kw in calls] == ["none", "none"]


def test_check_moe_plans_selects_gfx950_kernels(tmp_path):
    """On gfx950 (placeholder or real) the selection matches the served kernels."""
    from tokenspeed.cli._argsplit import _ensure_parser_platform, check_moe_plans

    platform = _ensure_parser_platform()
    if not platform.endswith("gfx950") and "-9.5" not in platform:
        pytest.skip(f"selection is arch-specific; platform {platform}")
    model = _v41_model_dir(tmp_path)
    r = check_moe_plans(_layout_argv(model, 1, 4, "--moe-backend", "auto"))["moe_plan"]
    # r0002-f3 tp4ep4 server log: "Kernel warmup covered draft experts
    # model.layers.0.ffn (gluon_mxfp4_a16w4_swiglu_ep_precomputed_moe_apply, 128 experts)"
    assert r["draft"]["apply_kernel_name"] == (
        "gluon_mxfp4_a16w4_swiglu_ep_precomputed_moe_apply"
    )
    r = check_moe_plans(
        _layout_argv(
            model,
            1,
            4,
            "--moe-backend",
            "gluon_petit",
            "--draft-moe-backend",
            "gluon_petit",
            "--all2all-backend",
            "gluon_petit",
        )
    )["moe_plan"]
    # round-1 EP4 Petit log: "(gluon_petit_mxfp4_megamoe_apply, 128 experts)"
    assert r["draft"]["apply_kernel_name"] == "gluon_petit_mxfp4_megamoe_apply"

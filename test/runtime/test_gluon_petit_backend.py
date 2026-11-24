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

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from transformers import GptOssConfig, PretrainedConfig

from tokenspeed.runtime.configs import model_config as model_config_module
from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41TextConfig
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.model_config import (
    PetitMoeShape,
    _apply_gluon_petit_chunk,
    petit_moe_shapes,
)
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.utils import (
    All2AllBackend,
    MoeBackend,
    decode_tokens_per_rank,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.models.gpt_oss import GptOssDecoderLayer
from tokenspeed.runtime.utils.server_args import ServerArgs


def _compressed_mxfp4_config(quant_format: str) -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "format": quant_format,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": None,
                }
            },
        }
    )


def _mapping() -> SimpleNamespace:
    return SimpleNamespace(
        nnodes=1,
        world_size=8,
        moe=SimpleNamespace(ep_size=8, tp_size=1),
        attn=SimpleNamespace(tp_size=1, dp_size=8),
        dense=SimpleNamespace(tp_size=1),
    )


def _validation_args(
    *,
    moe_backend: str,
    draft_moe_backend: str | None,
    all2all_backend: str,
    speculative_algorithm: str | None,
    max_num_seqs: int,
    dtype: str,
    chunked_prefill_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda",
        disable_prefill_graph=False,
        disable_pdl=False,
        moe_backend=moe_backend,
        draft_moe_backend=draft_moe_backend,
        all2all_backend=all2all_backend,
        mapping=_mapping(),
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location=None,
        speculative_algorithm=speculative_algorithm,
        speculative_num_draft_tokens=1,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=1024,
    )


def test_gluon_petit_backend_enums() -> None:
    assert All2AllBackend("gluon_petit").is_gluon_petit()
    assert MoeBackend("gluon_petit").is_gluon_petit()
    assert not MoeBackend("gluon_petit").is_mega_moe()


def test_gluon_petit_cli_backend_names() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "openai/gpt-oss-120b",
            "--moe-backend",
            "gluon_petit",
            "--draft-moe-backend",
            "gluon_petit",
            "--all2all-backend",
            "gluon_petit",
        ]
    )
    assert MoeBackend(args.moe_backend) is MoeBackend.GLUON_PETIT
    assert MoeBackend(args.draft_moe_backend) is MoeBackend.GLUON_PETIT
    assert All2AllBackend(args.all2all_backend) is All2AllBackend.GLUON_PETIT


class _PetitMoE(nn.Module):
    """Stands in for Petit's MegaMoE, which needs gfx950 and process groups."""

    owns_ep_communication = True

    def forward(self, hidden_states, num_global_tokens, max_num_tokens_per_gpu):
        return hidden_states * num_global_tokens + max_num_tokens_per_gpu


def test_gluon_petit_gpt_oss_moe_takes_rank_local_rows() -> None:
    mapping = Mapping(
        rank=0, world_size=2, attn_tp_size=1, attn_dp_size=2, moe_ep_size=2
    )
    layer = GptOssDecoderLayer.__new__(GptOssDecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = _PetitMoE()
    layer.comm_manager = CommManager(
        mapping=mapping,
        layer_id=1,
        is_moe=True,
        prev_is_moe=True,
        dense_batch_invariant=False,
        query_sharded=False,
    )
    hidden_states = torch.randn(3, 8)

    # With no forward context or process group, any CommManager leg would raise.
    out = layer.forward_mlp(hidden_states, hidden_states, None, 8, 5)

    torch.testing.assert_close(out, hidden_states * 8 + 5, atol=0, rtol=0)


def test_gluon_petit_requires_matching_backend_pair() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="none",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="--all2all-backend gluon_petit"):
        ServerArgs.validate_petit_moe_options(args)


def test_gluon_petit_rejects_mixed_draft_backend() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend="triton",
        all2all_backend="gluon_petit",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="incompatible draft=triton"):
        ServerArgs.validate_petit_moe_options(args)


def test_gluon_petit_rejects_non_bfloat16_dtype() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="gluon_petit",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="float16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="requires --dtype bfloat16"):
        ServerArgs.validate_petit_moe_options(args)


def test_dsv4_gluon_petit_zero_token_routing_shapes() -> None:
    from tokenspeed.runtime.models.deepseek_v41_moe import DeepseekV4MoE

    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.config = SimpleNamespace(num_experts_per_tok=6, n_routed_experts=384)
    hidden_states = torch.empty((0, 7168), dtype=torch.bfloat16)

    layer.gate = SimpleNamespace(tid2eid=None, e_score_correction_bias=None)
    scores, correction, hashes, input_ids = layer._routing_inputs(
        hidden_states, input_ids=None
    )

    assert scores.shape == (0, 384)
    assert scores.dtype == torch.float32
    assert correction is None
    assert hashes is None
    assert input_ids is None


@pytest.fixture
def petit_args() -> SimpleNamespace:
    return _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="gluon_petit",
        speculative_algorithm=None,
        max_num_seqs=8192,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )


def test_validate_calls_petit_validation() -> None:
    # Construct real ServerArgs without model/device initialization, then check
    # that the public validation entry point reaches the backend-pair check.
    with mock.patch.object(ServerArgs, "__post_init__", return_value=None):
        args = ServerArgs(model="test")
    args.moe_backend = "gluon_petit"
    args.all2all_backend = "none"
    with pytest.raises(ValueError, match="--all2all-backend gluon_petit"):
        args.validate()


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({}, None),
        ({"draft_moe_backend": "triton"}, None),  # Inactive draft is ignored.
        ({"speculative_algorithm": "MTP"}, None),  # Draft inherits target.
        ({"speculative_algorithm": "MTP", "draft_moe_backend": "gluon_petit"}, None),
        ({"moe_backend": "triton"}, "incompatible target=triton"),
        (
            {"moe_backend": "triton", "all2all_backend": "none", "dtype": "float16"},
            None,
        ),
        # The kernel checks capacity against the model's workspace, not here.
        ({"max_num_seqs": 8200}, None),
        ({"chunked_prefill_size": 8192, "max_prefill_tokens": 8192}, None),
        ({"chunked_prefill_size": 0}, "sizes its workspace from --chunked-prefill"),
    ],
)
def test_petit_shared_options(petit_args, overrides, error) -> None:
    vars(petit_args).update(overrides)
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        ServerArgs.validate_petit_moe_options(petit_args)


@pytest.mark.parametrize(
    "speculative_algorithm,expected",
    [
        ("DSPARK", 192),  # 128 requests / 4 ranks x 6 verify tokens.
        (None, 32),  # One token per request without speculation.
    ],
)
def test_decode_tokens_per_rank(speculative_algorithm, expected) -> None:
    assert decode_tokens_per_rank(128, 4, speculative_algorithm, 6) == expected


@pytest.mark.parametrize("attn_tp,dense_tp", [(2, 1), (1, 2)])
def test_petit_shared_parallelism(petit_args, attn_tp, dense_tp) -> None:
    petit_args.mapping.attn.tp_size = attn_tp
    petit_args.mapping.dense.tp_size = dense_tp
    with pytest.raises(ValueError, match="attention TP1 and dense TP1"):
        ServerArgs.validate_petit_moe_options(petit_args)


@pytest.mark.parametrize(
    "mapping_overrides,moe_overrides,options,layer_overrides,is_cdna4,error",
    [
        ({}, {}, {}, {}, True, None),
        (
            {},
            {},
            {},
            {
                "top_k": 16,
                "num_experts": 896,
                "hidden_size": 3584,
                "intermediate_size": 3072,
                "activation": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "routing_mode": "precomputed_topk",
                "quant_config": _compressed_mxfp4_config("mxfp4-pack-quantized"),
            },
            True,
            None,
        ),
        (
            {},
            {},
            {},
            {"quant_config": _compressed_mxfp4_config("pack-quantized")},
            True,
            "serialized MXFP4 expert weights",
        ),
        (
            {},
            {},
            {},
            {
                "quant_config": Mxfp4Config(
                    ignored_layers=[], is_checkpoint_mxfp4_serialized=False
                )
            },
            True,
            "serialized MXFP4 expert weights",
        ),
        ({}, {}, {"init_expert_location": "trivial"}, {}, True, None),
        ({}, {}, {}, {}, False, "requires AMD CDNA4"),
        (
            {},
            {},
            {"moe_mxfp4_fp8_activation": True},
            {},
            True,
            "requested fp8 activations are unsupported",
        ),
        ({"nnodes": 2}, {}, {}, {}, True, "supports one node only"),
        ({"world_size": 4}, {}, {}, {}, True, "world_size=ep_size in"),
        ({}, {"ep_size": 4}, {}, {}, True, "world_size=ep_size in"),
        (
            {"world_size": 2},
            {"ep_size": 2},
            {},
            {"ep_size": 2},
            True,
            "world_size=ep_size in",
        ),
        (
            {"world_size": 16},
            {"ep_size": 16},
            {},
            {"ep_size": 16},
            True,
            "world_size=ep_size in",
        ),
        ({"world_size": 4}, {"ep_size": 4}, {}, {"ep_size": 4}, True, None),
        ({}, {"tp_size": 2}, {}, {}, True, "MoE tensor parallel size 1"),
        ({}, {}, {}, {"ep_size": 4}, True, "world_size=ep_size in"),
        ({}, {}, {}, {"ep_size": 1, "tp_size": 2}, True, "MoE tensor parallel size 1"),
        ({}, {}, {"enable_eplb": True}, {}, True, "trivial expert placement"),
        ({}, {}, {"ep_num_redundant_experts": 8}, {}, True, "trivial expert placement"),
        (
            {},
            {},
            {"init_expert_location": "custom"},
            {},
            True,
            "trivial expert placement",
        ),
        ({}, {}, {}, {"quant_config": None}, True, "serialized MXFP4 expert weights"),
        ({}, {}, {}, {"activation_alpha": 1.5}, True, "nonstandard SiLU alpha"),
    ],
)
def test_petit_layer_constraints(
    petit_args,
    mapping_overrides,
    moe_overrides,
    options,
    layer_overrides,
    is_cdna4,
    error,
) -> None:
    vars(petit_args.mapping).update(mapping_overrides)
    vars(petit_args.mapping.moe).update(moe_overrides)
    vars(petit_args).update(options)
    # These restrictions belong to the layer, so server validation accepts them.
    with mock.patch(
        "tokenspeed.runtime.utils.server_args.current_platform",
        return_value=SimpleNamespace(is_cdna4=is_cdna4),
    ):
        ServerArgs.validate_petit_moe_options(petit_args)
    layer_args = dict(
        top_k=4,
        num_experts=128,
        hidden_size=2880,
        intermediate_size=3072,
        quant_config=Mxfp4Config(
            ignored_layers=[], is_checkpoint_mxfp4_serialized=True
        ),
        layer_index=0,
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=8,
    )
    layer_args.update(layer_overrides)
    with (
        mock.patch.dict(
            expert_module.global_server_args_dict,
            mapping=petit_args.mapping,
            enable_eplb=petit_args.enable_eplb,
            ep_num_redundant_experts=petit_args.ep_num_redundant_experts,
            init_expert_location=petit_args.init_expert_location,
            moe_mxfp4_fp8_activation=options.get("moe_mxfp4_fp8_activation", False),
            # Below the decode batch, so the capacity must take the decode term.
            chunked_prefill_size=256,
            max_num_seqs=petit_args.max_num_seqs,
            speculative_algorithm=petit_args.speculative_algorithm,
            speculative_num_draft_tokens=petit_args.speculative_num_draft_tokens,
        ),
        mock.patch.object(
            expert_module,
            "current_platform",
            return_value=SimpleNamespace(is_cdna4=is_cdna4),
        ),
        mock.patch.object(
            expert_module,
            "get_all2all_backend",
            return_value=All2AllBackend.GLUON_PETIT,
        ),
        mock.patch.object(
            expert_module, "get_moe_backend", return_value=MoeBackend.GLUON_PETIT
        ),
        mock.patch.object(
            expert_module,
            "kernel_moe_plan",
            return_value={"solution": "gluon"},
        ) as plan,
        mock.patch.object(expert_module, "create_layer_weights") as weights,
    ):
        with pytest.raises(ValueError, match=error) if error else nullcontext():
            MoELayer(**layer_args)
        if error:
            plan.assert_not_called()
            weights.assert_not_called()
        else:
            plan.assert_called_once()
            assert plan.call_args.kwargs["ep_size"] == layer_args["ep_size"]
            assert plan.call_args.kwargs["solution"] == "gluon"
            assert plan.call_args.kwargs["a2a_backend"] == "gluon_petit"
            assert plan.call_args.kwargs["activation"] == layer_args.get(
                "activation", "silu"
            )
            # max(256-token chunk, 8192 requests / 8 attention-DP ranks).
            assert plan.call_args.kwargs["persistent_max_num_tokens_per_gpu"] == 1024
            weights.assert_called_once()


def _model_config(architecture: str, text: PretrainedConfig) -> SimpleNamespace:
    # A ModelConfig needs the checkpoint; the shapes read only these two.
    return SimpleNamespace(
        hf_config=PretrainedConfig(architectures=[architecture]),
        hf_text_config=text,
    )


# DeepseekV41TextConfig has no defaults; these are the DeepSeek-V4.1-Flash
# checkpoint's text_config values.
_DSV41_FLASH = _model_config(
    "DeepseekV41ForCausalLM",
    DeepseekV41TextConfig(
        n_routed_experts=384,
        num_experts_per_tok=6,
        hidden_size=5120,
        moe_intermediate_size=2304,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
    ),
)


def _chunk_args(*, ep_size: int, chunk: int | None = None, **overrides):
    args = SimpleNamespace(
        all2all_backend="gluon_petit",
        mapping=SimpleNamespace(
            moe=SimpleNamespace(ep_size=ep_size),
            attn=SimpleNamespace(dp_size=ep_size),
        ),
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        max_num_seqs=128,
        chunked_prefill_size=8192 if chunk is None else chunk,
        _chunked_prefill_size_unset=chunk is None,
    )
    vars(args).update(overrides)
    return args


@pytest.fixture
def cdna4():
    with mock.patch.object(
        model_config_module,
        "current_platform",
        return_value=SimpleNamespace(is_cdna4=True),
    ):
        yield


@pytest.mark.parametrize(
    "ep_size,overrides,expected",
    [
        # The DSv4.1 Flash workspace passes 32-bit offsets above 7808 rows at
        # EP4 and 2624 at EP8; its DSpark draft fits more at both.
        (4, {}, 7808),
        (8, {}, 2624),
        # A server without Petit keeps the default.
        (4, {"all2all_backend": "none"}, 8192),
    ],
)
def test_petit_caps_an_unset_chunk_at_its_capacity(cdna4, ep_size, overrides, expected):
    args = _chunk_args(ep_size=ep_size, **overrides)
    _apply_gluon_petit_chunk(_DSV41_FLASH, args)
    assert args.chunked_prefill_size == expected


def test_petit_leaves_the_chunk_off_cdna4() -> None:
    # Off CDNA4 the MoE layer refuses Petit; the chunk does not consult it.
    args = _chunk_args(ep_size=4)
    with mock.patch.object(
        model_config_module,
        "current_platform",
        return_value=SimpleNamespace(is_cdna4=False),
    ):
        _apply_gluon_petit_chunk(_DSV41_FLASH, args)
    assert args.chunked_prefill_size == 8192


@pytest.mark.parametrize("chunk", [4096, 7808])
def test_petit_keeps_an_explicit_chunk_that_fits(cdna4, chunk) -> None:
    args = _chunk_args(ep_size=4, chunk=chunk)
    _apply_gluon_petit_chunk(_DSV41_FLASH, args)
    assert args.chunked_prefill_size == chunk


def test_petit_refuses_an_explicit_chunk_above_its_capacity(cdna4) -> None:
    with pytest.raises(
        ValueError,
        match="--chunked-prefill-size 8192: Gluon Petit MegaMoE fits at most "
        "7808 tokens per rank for dsv41_flash at EP4",
    ):
        _apply_gluon_petit_chunk(_DSV41_FLASH, _chunk_args(ep_size=4, chunk=8192))


def test_petit_refuses_a_decode_batch_above_its_capacity(cdna4) -> None:
    # 8192 requests over 4 ranks x 6 verify tokens = 12288 rows per rank.
    args = _chunk_args(ep_size=4, max_num_seqs=8192)
    with pytest.raises(ValueError, match="decode batch needs 12288.*--max-num-seqs"):
        _apply_gluon_petit_chunk(_DSV41_FLASH, args)


@pytest.mark.parametrize(
    "model_config,ep_size,profile",
    [
        (
            _model_config(
                "DeepseekV41ForCausalLM",
                DeepseekV41TextConfig(
                    n_routed_experts=384,
                    num_experts_per_tok=6,
                    hidden_size=7168,
                    moe_intermediate_size=3072,
                ),
            ),
            8,
            "dsv4",
        ),
        (_DSV41_FLASH, 4, "dsv41_flash"),
        # Kimi K3's and GPT-OSS's config classes default to their checkpoints.
        (
            _model_config("KimiK3ForConditionalGeneration", KimiLinearConfig()),
            8,
            "kimi_k3",
        ),
        (_model_config("GptOssForCausalLM", GptOssConfig()), 8, "gpt_oss_120b"),
        # MuseSparkConfig requires every field; it copies these from text_config.
        (
            _model_config(
                "MuseSparkForConditionalGeneration",
                PretrainedConfig(
                    num_local_experts=256,
                    num_experts_per_tok=8,
                    moe_hidden_size=4096,
                    moe_intermediate_size=4096,
                ),
            ),
            8,
            "muse_spark",
        ),
    ],
)
def test_petit_moe_shapes_name_a_petit_profile(model_config, ep_size, profile):
    from tokenspeed_kernel.ops.moe.gluon import gluon_petit_capacity

    (shape,) = petit_moe_shapes(model_config, speculative_algorithm=None)
    capacity = gluon_petit_capacity(
        shape.num_experts,
        shape.top_k,
        shape.hidden_size,
        shape.intermediate_size,
        ep_size,
    )
    assert capacity is not None and capacity.profile == profile


def test_petit_moe_shapes_add_the_dspark_draft() -> None:
    assert petit_moe_shapes(_DSV41_FLASH, "DSPARK") == (
        PetitMoeShape(384, 6, 5120, 2304),
        PetitMoeShape(128, 3, 5120, 2304),
    )
    assert (
        petit_moe_shapes(_model_config("LlamaForCausalLM", PretrainedConfig()), None)
        is None
    )

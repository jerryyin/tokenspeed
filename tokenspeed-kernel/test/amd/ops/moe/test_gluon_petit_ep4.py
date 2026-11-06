# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT
"""CPU-only checks for the Gluon Petit EP world-size generalisation {4, 8}.

No GPU kernel is launched: MegaMoeConfig construction only encodes a solution
id, and the workspace path is exercised with a mocked ``petit_kernel`` and a
mocked ``torch.distributed``.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.gluon import petit as adapter
from tokenspeed_kernel.ops.moe.gluon.petit import (
    _DSV4_PROFILE,
    _DSV41_FLASH_DSPARK_PROFILE,
    _DSV41_FLASH_PROFILE,
    _GPT_OSS_120B_PROFILE,
    _KIMI_K3_PROFILE,
    SUPPORTED_WORLD_SIZES,
    _validate_layer,
)
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.thirdparty.gluon_petit import petit_kernel

V41_PROFILES = [_DSV41_FLASH_PROFILE, _DSV41_FLASH_DSPARK_PROFILE]
EP8_ONLY_PROFILES = [_GPT_OSS_120B_PROFILE, _DSV4_PROFILE, _KIMI_K3_PROFILE]

# Solution ids computed at the unpatched baseline (31021d0f + idle-forward
# hunk) for world_size=8; the patch must keep EP8 bit-identical.
EP8_BASELINE_SOLUTION_IDS = {
    "dsv41_flash": 0x2141AFB18011,
    "dsv41_flash_dspark": 0x2140CFB18011,
}


def _v41_config(profile, world_size):
    return petit_kernel.MegaMoeConfig(
        world_size=world_size,
        num_experts=profile.num_experts,
        topk=profile.top_k,
        model_dim=profile.model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction.silu_clamped,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=profile.inter_dim,
        has_bias=False,
    )


def test_supported_world_sizes_constant() -> None:
    assert SUPPORTED_WORLD_SIZES == (4, 8)
    assert petit_kernel.DEEPSEEK_V41_WORLD_SIZES == (4, 8)


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("world_size", [4, 8])
def test_v41_megamoe_config_registered(profile, world_size) -> None:
    config = _v41_config(profile, world_size)
    assert config.world_size == world_size
    # Rank count is encoded as log2(world_size) in bits 24-25 of the solution id.
    assert (config._solution_id >> 24) & 0x3 == world_size.bit_length() - 1


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
def test_v41_ep4_and_ep8_ids_differ_only_in_rank_field(profile) -> None:
    id4 = _v41_config(profile, 4)._solution_id
    id8 = _v41_config(profile, 8)._solution_id
    assert id4 != id8
    assert id4 & ~(0x3 << 24) == id8 & ~(0x3 << 24)


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
def test_v41_ep8_solution_id_matches_baseline(profile) -> None:
    expected = EP8_BASELINE_SOLUTION_IDS.get(profile.name)
    if expected is None:
        pytest.skip("baseline id not recorded")
    assert _v41_config(profile, 8)._solution_id == expected


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("world_size", [1, 2, 16])
def test_v41_megamoe_config_rejects_other_world_sizes(profile, world_size) -> None:
    with pytest.raises(ValueError):
        _v41_config(profile, world_size)


def test_dsv4_megamoe_config_stays_ep8_only() -> None:
    kwargs = dict(
        num_experts=384,
        topk=6,
        model_dim=7168,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction.silu,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=False,
    )
    petit_kernel.MegaMoeConfig(world_size=8, **kwargs)
    with pytest.raises(ValueError, match="unsupported registered"):
        petit_kernel.MegaMoeConfig(world_size=4, **kwargs)


def _v41_layer(profile, ep_size, num_local_experts=None):
    return SimpleNamespace(
        num_experts=profile.num_experts,
        top_k=profile.top_k,
        hidden_size=profile.model_dim,
        intermediate_size=profile.logical_intermediate,
        ep_size=ep_size,
        tp_size=1,
        num_local_experts=(
            profile.num_experts // ep_size
            if num_local_experts is None
            else num_local_experts
        ),
        activation="silu",
        swiglu_beta=None,
        swiglu_arg=SimpleNamespace(alpha=None, limit=10.0),
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("ep_size", [4, 8])
def test_validate_layer_accepts_v41_ep4_and_ep8(profile, ep_size) -> None:
    assert _validate_layer(_v41_layer(profile, ep_size)) is profile


def test_v41_local_expert_counts() -> None:
    # 384 experts: 96 per rank at EP4, 48 at EP8. DSpark 128: 32 / 16.
    assert _v41_layer(_DSV41_FLASH_PROFILE, 4).num_local_experts == 96
    assert _v41_layer(_DSV41_FLASH_PROFILE, 8).num_local_experts == 48
    assert _v41_layer(_DSV41_FLASH_DSPARK_PROFILE, 4).num_local_experts == 32
    assert _v41_layer(_DSV41_FLASH_DSPARK_PROFILE, 8).num_local_experts == 16


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("ep_size", [1, 2, 16])
def test_validate_layer_rejects_unsupported_ep(profile, ep_size) -> None:
    with pytest.raises(ValueError, match="supports EP sizes"):
        _validate_layer(_v41_layer(profile, ep_size))


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
def test_validate_layer_rejects_ep8_shard_count_at_ep4(profile) -> None:
    layer = _v41_layer(profile, 4, num_local_experts=profile.num_experts // 8)
    with pytest.raises(ValueError, match="contiguous expert shards"):
        _validate_layer(layer)


def test_validate_layer_keeps_dsv4_ep8_only() -> None:
    layer = SimpleNamespace(
        num_experts=384,
        top_k=6,
        hidden_size=7168,
        intermediate_size=3072,
        ep_size=4,
        tp_size=1,
        num_local_experts=96,
        activation="swiglu",
        swiglu_beta=None,
        swiglu_arg=SimpleNamespace(alpha=None, limit=None),
        w13_input_layout="concatenated",
        w13_weight_bias=None,
        w2_weight_bias=None,
    )
    with pytest.raises(ValueError, match=r"supports EP sizes \(8,\)"):
        _validate_layer(layer)


@pytest.mark.parametrize("profile", V41_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("world_size", [4, 8])
def test_get_workspace_uses_actual_world_size(profile, world_size) -> None:
    fake_kernel = mock.MagicMock()
    fake_kernel.MegaMoeActivationFunction = petit_kernel.MegaMoeActivationFunction
    adapter._workspace_cache.clear()
    try:
        with (
            mock.patch.object(adapter, "_import_petit_kernel", return_value=fake_kernel),
            mock.patch.object(adapter.dist, "is_initialized", return_value=True),
            mock.patch.object(adapter.dist, "get_world_size", return_value=world_size),
        ):
            ws = adapter._get_workspace(torch.device("cuda", 0), profile)
            again = adapter._get_workspace(torch.device("cuda", 0), profile)
        assert ws is again
        assert fake_kernel.MegaMoeConfig.call_args.kwargs["world_size"] == world_size
        fake_kernel.create_vmm_symmetric_heap.assert_called_once_with(world_size)
        fake_kernel.MegaMoeConfig.return_value.input_views.assert_called_once_with(
            fake_kernel.create_vmm_symmetric_heap.return_value, 1024
        )
        assert (0, profile) in adapter._workspace_cache
    finally:
        adapter._workspace_cache.clear()


@pytest.mark.parametrize(
    "profile,world_size",
    [(_DSV41_FLASH_PROFILE, 2), (_DSV41_FLASH_PROFILE, 16), (_DSV4_PROFILE, 4)],
)
def test_get_workspace_rejects_unsupported_world_size(profile, world_size) -> None:
    adapter._workspace_cache.clear()
    with (
        mock.patch.object(adapter, "_import_petit_kernel") as imp,
        mock.patch.object(adapter.dist, "is_initialized", return_value=True),
        mock.patch.object(adapter.dist, "get_world_size", return_value=world_size),
        pytest.raises(RuntimeError, match="world size in"),
    ):
        adapter._get_workspace(torch.device("cuda", 0), profile)
    imp.assert_not_called()


def _plan(ep_size, mi350_platform):
    profile = _DSV41_FLASH_PROFILE
    with mock.patch(
        "tokenspeed_kernel.selection.current_platform", return_value=mi350_platform
    ):
        return moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode="precomputed_topk",
            a2a_backend="gluon_petit",
            ep_size=ep_size,
            ispp=profile.logical_intermediate,
            hidden=profile.model_dim,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            internal_activation_dtype="mxfp4",
            with_bias=False,
            fast_math=True,
            combine_order="rank",
            solution="gluon",
        )


@pytest.mark.parametrize("ep_size", [4, 8])
def test_plan_selects_petit_for_ep4_and_ep8(ep_size, mi350_platform) -> None:
    plan = _plan(ep_size, mi350_platform)
    assert plan["apply_kernel_name"] == "gluon_petit_mxfp4_megamoe_apply"


@pytest.mark.parametrize("ep_size", [2, 16])
def test_plan_rejects_other_ep_sizes(ep_size, mi350_platform) -> None:
    with pytest.raises(NoKernelFoundError):
        _plan(ep_size, mi350_platform)

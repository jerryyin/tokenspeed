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

"""Off-device source contracts for gfx1250 MXFP4 TDM index ownership."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MXFP4_ROOT = (
    REPO_ROOT
    / "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/moe/mxfp4"
)
COMMON = MXFP4_ROOT / "_common.py"
INDEX_LAYOUT_CONSUMERS = (COMMON, MXFP4_ROOT / "decode.py", MXFP4_ROOT / "fused.py")
INDEXING = MXFP4_ROOT / "_indexing.py"
FUSED = MXFP4_ROOT / "fused.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _index_layout_function() -> ast.FunctionDef:
    for node in _parse(COMMON).body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "get_tdm_gather_scatter_idx_layout"
        ):
            return node
    raise AssertionError("get_tdm_gather_scatter_idx_layout is missing")


def _index_layout_return() -> ast.Call:
    returns = [
        node for node in _index_layout_function().body if isinstance(node, ast.Return)
    ]
    assert len(returns) == 1
    call = returns[0].value
    assert isinstance(call, ast.Call)
    return call


def test_index_layout_partitions_rows_across_all_warps() -> None:
    function = _index_layout_function()
    assertions = {
        ast.unparse(node.test) for node in function.body if isinstance(node, ast.Assert)
    }
    assert assertions == {
        "NUM_WARPS > 0",
        "NUM_INDICES % NUM_WARPS == 0",
    }

    expected = ast.parse(
        "gl.BlockedLayout("
        "[1, NUM_INDICES // NUM_WARPS], "
        "[32, 1], [1, NUM_WARPS], [0, 1])",
        mode="eval",
    ).body
    assert ast.dump(_index_layout_return(), include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )

    num_indices = 16
    num_warps = 8
    rows_per_warp = num_indices // num_warps
    owners = {
        warp: tuple(range(warp * rows_per_warp, (warp + 1) * rows_per_warp))
        for warp in range(num_warps)
    }
    assert owners == {
        0: (0, 1),
        1: (2, 3),
        2: (4, 5),
        3: (6, 7),
        4: (8, 9),
        5: (10, 11),
        6: (12, 13),
        7: (14, 15),
    }


def test_every_index_layout_consumer_keeps_the_warp_distributed_axis() -> None:
    found = {}
    for path in INDEX_LAYOUT_CONSUMERS:
        calls = []
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call) or len(node.args) != 2:
                continue
            if (
                not isinstance(node.func, ast.Attribute)
                or node.func.attr != "SliceLayout"
            ):
                continue
            if (
                not isinstance(node.args[1], ast.Name)
                or node.args[1].id != "IDX_BASE_LAYOUT"
            ):
                continue
            calls.append(node)

        assert len(calls) == 1, path
        slice_dim = calls[0].args[0]
        assert isinstance(slice_dim, ast.Constant)
        found[path.name] = slice_dim.value

    assert found == {"_common.py": 0, "decode.py": 0, "fused.py": 0}


def test_index_layout_is_not_the_two_axis_warp_split_shortcut() -> None:
    call = _index_layout_return()
    actual_warps_per_cta = ast.unparse(call.args[2])
    naive_warps_per_cta = ast.unparse(
        ast.parse("[2, NUM_WARPS // 2]", mode="eval").body
    )

    assert actual_warps_per_cta == "[1, NUM_WARPS]"
    assert actual_warps_per_cta != naive_warps_per_cta


def test_tdm_index_width_uses_full_domain_and_scatter_sentinel() -> None:
    spec = importlib.util.spec_from_file_location("gfx1250_mxfp4_indexing", INDEXING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    select = module.select_tdm_index_width_bits

    assert select(gather_input_rows=None, scatter_writeback_rows=None) == 32
    assert select(gather_input_rows=65_536, scatter_writeback_rows=None) == 16
    assert select(gather_input_rows=65_537, scatter_writeback_rows=None) == 32
    assert select(gather_input_rows=None, scatter_writeback_rows=65_535) == 16
    assert select(gather_input_rows=None, scatter_writeback_rows=65_536) == 32
    assert select(gather_input_rows=65_536, scatter_writeback_rows=65_535) == 16
    assert select(gather_input_rows=65_537, scatter_writeback_rows=1) == 32

    for kwargs in (
        {"gather_input_rows": -1, "scatter_writeback_rows": None},
        {"gather_input_rows": None, "scatter_writeback_rows": -1},
    ):
        try:
            select(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"negative index domain was accepted: {kwargs}")


def test_host_width_selection_reaches_the_kernel_constexpr() -> None:
    tree = _parse(FUSED)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    selector = functions["get_index_type"]
    wrapper = functions["matmul"]

    selector_source = ast.unparse(selector)
    assert "int(a.shape_max[-2]) if isinstance(a, Tensor) else int(a.shape[-2])" in (
        selector_source
    )
    assert (
        "None if scatter_indx is None else int(scatter_indx.shape[0])"
        in selector_source
    )
    assert selector_source.count("select_tdm_index_width_bits(") == 1

    assignments = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "index_type"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "get_index_type(a, gather_indx, scatter_indx)"

    launches = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "INDEX_TYPE" for keyword in node.keywords)
    ]
    assert len(launches) == 1
    index_keywords = [
        keyword for keyword in launches[0].keywords if keyword.arg == "INDEX_TYPE"
    ]
    assert len(index_keywords) == 1
    assert ast.unparse(index_keywords[0].value) == "index_type"

    # The rejected shortcut is a hard-coded narrow launch type. It would make
    # small campaign fixtures pass while silently truncating a larger domain.
    assert "INDEX_TYPE=gl.int16" not in FUSED.read_text(encoding="utf-8")


def test_descriptor_index_width_is_separate_from_address_width() -> None:
    common = _parse(COMMON)
    descriptor_casts = [
        node
        for node in ast.walk(common)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and ast.unparse(node.func.value).startswith("gl.load(")
    ]
    assert len(descriptor_casts) == 1
    assert ast.unparse(descriptor_casts[0].args[0]) == "cfg.index_type"

    for path in (MXFP4_ROOT / "decode.py", MXFP4_ROOT / "fused.py"):
        tree = _parse(path)
        source = path.read_text(encoding="utf-8")
        annotated_names = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert "address_index_type: gl.constexpr" in source
        assert "index_type=INDEX_TYPE" in source
        assert "address_index_type" in annotated_names
        assert "index_type" not in annotated_names


def test_no_gather_descriptor_extent_cast_is_surgical() -> None:
    expected_extent = ast.parse("(eM - off_m).to(gl.int32)", mode="eval").body
    expected_address_type = ast.parse(
        "gl.int64 if UPCAST_INDICES else gl.int32", mode="eval"
    ).body

    for path in (MXFP4_ROOT / "decode.py", MXFP4_ROOT / "fused.py"):
        tree = _parse(path)
        source = path.read_text(encoding="utf-8")

        address_types = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "address_index_type"
        ]
        assert len(address_types) == 1, path
        assert ast.dump(address_types[0], include_attributes=False) == ast.dump(
            expected_address_type, include_attributes=False
        )

        extent_casts = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "descriptor_m"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        ]
        assert len(extent_casts) == 1, path
        assert ast.dump(extent_casts[0], include_attributes=False) == ast.dump(
            expected_extent, include_attributes=False
        )

        # The rejected shortcut is to narrow all indices or the expert base
        # pointer.  Those values must continue to use address_index_type.
        assert (
            "expt_id, off_m = expt_id.to(address_index_type), off_m.to(address_index_type)"
            in source
        )
        assert "W_ptr = W + expt_id * stride_w_e" in source
        assert "expt_id.to(gl.int32)" not in source
        assert "off_m.to(gl.int32)" not in source

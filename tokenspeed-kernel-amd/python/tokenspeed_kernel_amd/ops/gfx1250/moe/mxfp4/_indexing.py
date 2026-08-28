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

"""Host-side index-width selection for gfx1250 TDM gather and scatter."""

from __future__ import annotations


def select_tdm_index_width_bits(
    *,
    gather_input_rows: int | None,
    scatter_writeback_rows: int | None,
) -> int:
    """Return the narrowest safe gfx1250 TDM index width.

    Gather values range from zero through ``gather_input_rows - 1``. Scatter
    additionally uses ``scatter_writeback_rows`` itself as the masked-off
    sentinel, so that value must also fit.
    """

    if gather_input_rows is None and scatter_writeback_rows is None:
        return 32
    if gather_input_rows is not None:
        if gather_input_rows < 0:
            raise ValueError("gather_input_rows must be nonnegative")
        if gather_input_rows > 1 << 16:
            return 32
    if scatter_writeback_rows is not None:
        if scatter_writeback_rows < 0:
            raise ValueError("scatter_writeback_rows must be nonnegative")
        if scatter_writeback_rows > (1 << 16) - 1:
            return 32
    return 16

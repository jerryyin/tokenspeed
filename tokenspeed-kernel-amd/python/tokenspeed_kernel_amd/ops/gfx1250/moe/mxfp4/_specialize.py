"""Gluon closure specialization used by the donor-synchronized decode path.

This is the host-side specialization mechanism used by the clean gfx1250
donor.  Keeping it next to the transplanted kernel makes the compiler-visible
activation closure explicit while avoiding a runtime dependency on the donor
checkout.
"""

from __future__ import annotations

import inspect
import re
import textwrap
import types

from tokenspeed_kernel_amd._triton import gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.moe._common import FnSpecs
from triton.experimental.gluon._runtime import GluonJITFunction


class ClosureArg:
    def __init__(self, fn_name: str, fn_params_name: str):
        self.fn_name = fn_name
        self.fn_params_name = fn_params_name


def define_kernel(src, module, attrs=None, is_gluon=False, **extra_globals):
    """Define a JIT kernel from the donor specialization's generated source."""

    def _empty_fn():
        pass

    globals_dict = dict(_empty_fn.__globals__)
    globals_dict.update(extra_globals)
    function = types.FunctionType(_empty_fn.__code__, globals_dict)
    function.__module__ = module.__name__

    src = textwrap.dedent(src)
    src = src[src.find("def ") :]
    stored_functions = []
    function_name = src[4:].split("(")[0].strip()
    globals_dict["stored_functions"] = stored_functions
    exec(src + "\n\nstored_functions.append(" + function_name + ")\n", globals_dict)

    function.__signature__ = inspect.signature(stored_functions[0])
    function.__name__ = function_name
    function.__doc__ = stored_functions[0].__doc__
    attrs = {} if attrs is None else attrs
    function = (
        GluonJITFunction(function, **attrs)
        if is_gluon
        else triton.JITFunction(function, **attrs)
    )
    function._unsafe_update_src(src)
    return function


def specialize(fn, module, constants, tuples, name=None, do_not_specialize=()):
    """Specialize closure arguments exactly as the gfx1250 donor does."""

    assert isinstance(fn, triton.runtime.jit.JITFunction)
    if name is None:
        name = fn.__name__
    src = textwrap.dedent(inspect.getsource(fn.fn))
    lines = src.split("\n")
    def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def"))
    header_end = def_idx
    while not lines[header_end].rstrip().endswith(":"):
        header_end += 1
    body_lines = lines[header_end + 1 :]
    header_lines = lines[def_idx : header_end + 1]
    header_clean = [
        line.split("#", 1)[0].strip()
        for line in header_lines
        if line.split("#", 1)[0].strip()
    ]
    match = re.search(r"\((.*)\)\s*:", " ".join(header_clean))
    if match is None:
        raise ValueError("Could not parse function header")
    args = [arg.strip() for arg in match.group(1).split(",") if arg.strip()]
    non_specialized_args = []
    for arg in args:
        arg_key = arg.split(":")[0].split("=")[0].strip()
        replacement = tuples.get(arg_key, [arg])
        if arg_key not in constants:
            non_specialized_args.extend(replacement)

    specialized_functions = {
        value.__name__: value
        for value in constants.values()
        if isinstance(value, triton.runtime.jit.JITFunction)
    }
    generated_globals = specialized_functions | fn.get_capture_scope()
    new_signature = f"def {name}({', '.join(non_specialized_args)}):"
    language_module = "gl" if fn.is_gluon() else "tl"
    constexpr_lines = [
        f"    {key}: {language_module}.constexpr = "
        f"{value.__name__ if callable(value) else value}"
        for key, value in constants.items()
    ]
    tuple_lines = [
        f"    {key} = ({','.join(value)}{',' if len(value) >= 1 else ''})"
        for key, value in tuples.items()
    ]
    generated_src = "\n".join(
        ["@gluon.jit" if fn.is_gluon() else "@triton.jit", new_signature]
        + constexpr_lines
        + tuple_lines
        + body_lines
    )

    new_preamble_len = 1 + len(constexpr_lines) + len(tuple_lines)
    line_delta = new_preamble_len - len(header_lines)
    signature = inspect.signature(triton.runtime.jit.JITFunction.__init__)
    attrs = {
        parameter.name: getattr(fn, parameter.name, parameter.default)
        for parameter in list(signature.parameters.values())[2:]
    }
    base_repr = attrs["repr"]

    def new_repr(specialization):
        result = base_repr(specialization)
        for specialized_function in specialized_functions.values():
            specialized_repr = specialized_function.repr(None)
            if specialized_repr:
                specialized_repr = specialized_repr.rsplit(".", 1)[-1].strip("_")
            if specialized_repr:
                result += f"_{specialized_repr}"
        return result

    attrs["repr"] = new_repr
    if do_not_specialize:
        attrs["do_not_specialize"] = do_not_specialize
    result = define_kernel(
        generated_src,
        module,
        attrs,
        is_gluon=fn.is_gluon(),
        **generated_globals,
    )

    adjust_line_number = lambda line: max(1, line - line_delta)
    result.raw_src = list(fn.raw_src)
    result.starting_line_number = adjust_line_number(fn.starting_line_number)
    result.def_file_line_number = adjust_line_number(fn.def_file_line_number)
    result.def_file_col_number = fn.def_file_col_number
    original_code = fn.fn.__code__
    result.file_name = original_code.co_filename
    result.fn.__code__ = result.fn.__code__.replace(
        co_filename=original_code.co_filename,
        co_firstlineno=adjust_line_number(original_code.co_firstlineno),
    )
    return result


class SpecializationModule:
    def __init__(self, module_name: str, kernels, closure_args):
        self.module_name = module_name
        self.kernels = kernels
        self.closure_args = closure_args
        self._modules = {}

    def get(self, **kwargs):
        import sys

        specs = [FnSpecs.default()] * len(self.closure_args)
        for key, value in kwargs.items():
            specs[list(self.closure_args.keys()).index(key)] = value
        cache_key = tuple(spec.name for spec in specs)
        if cache_key in self._modules:
            return self._modules[cache_key]
        spec_constants = {
            arg.fn_name: spec.fn
            for arg, spec in zip(self.closure_args.values(), specs)
        }
        spec_tuples = {
            arg.fn_params_name: spec.fn_arg_names
            for arg, spec in zip(self.closure_args.values(), specs)
        }
        do_not_specialize = []
        for spec in specs:
            do_not_specialize.extend(spec.fn_arg_do_not_specialize)
        module = types.ModuleType(self.module_name + "_".join(cache_key))
        sys.modules[module.__name__] = module
        for kernel_name, kernel_fn in self.kernels:
            setattr(
                module,
                kernel_name,
                specialize(
                    kernel_fn,
                    module,
                    spec_constants,
                    spec_tuples,
                    do_not_specialize=do_not_specialize,
                ),
            )
        self._modules[cache_key] = module
        return module

import itertools

import unittest
import weakref
from typing import List, Optional, Tuple

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree

from torch._dynamo.utils import detect_fake_mode
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.fx_passes.freezing_patterns import get_freezing_patterns
from torch.ao.quantization._pt2e.utils import _fuse_conv_bn_
from torch.fx.experimental.proxy_tensor import make_fx
from . import config
from .decomposition import select_decomp_table

aten = torch.ops.aten


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0

    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


def replace_params_with_constants(gm, flat_params, fw_metadata) -> List[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.

    Returns a list of indices representing the input parameters that were not converted to constants.
    """

    params = [node for node in gm.graph.nodes if node.op == "placeholder"]
    fake_inp_nodes = params[: len(params)]

    g = gm.graph

    preserved_arg_indices = []
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]

    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in fw_metadata.mutated_inp_indices or aliased_input_args:
            preserved_arg_indices.append(i)
            continue

        replace_node_with_constant(gm, node, real_input)

    # add on non param inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))

    # is this necessary ?
    gm.recompile()
    return preserved_arg_indices


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm):
    unknown_value = object()

    node_replacements = {}

    class ConstantFolder(torch.fx.Interpreter):
        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)

            flattened_inputs = pytree.tree_flatten((args, kwargs))[0]
            if unknown_value in flattened_inputs:
                return unknown_value

            # skip constructors, since inductor generates optimal code for them already
            # and turning into tensor would result in an additional global memory read
            # TODO - more complicated strategy
            if node.op != "get_attr" and not any(
                isinstance(e, torch.Tensor) for e in flattened_inputs
            ):
                return unknown_value

            # All mutations should either be removed or on inputs which we did not make constant
            if (
                isinstance(node.target, torch._ops.OpOverload)
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                return unknown_value

            out = super().run_node(node)

            # TODO - remove constant from node_replacement when it has no uses
            if node.op != "get_attr" and isinstance(out, torch.Tensor):
                node_replacements[node] = out

            return out

        def run(self):
            env = {}
            for n in self.module.graph.nodes:
                if n.op == "placeholder":
                    env[n] = unknown_value
            return super().run(initial_env=env)

    ConstantFolder(gm).run()

    for node, constant in node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


@torch.utils._python_dispatch._disable_current_modes()
def fuse_conv_bn(gm):
    return _fuse_conv_bn_(gm)


def decompose_unfused_batchnorms(gm, example_inputs, preserved_arg_indices):
    if not any(
        node.target is aten._native_batch_norm_legit_no_training.default
        for node in gm.graph.nodes
    ):
        return gm

    fake_mode = detect_fake_mode(example_inputs)

    # constant params will be real tensors, not fake
    # TODO: fake_mode should should enable py dispatcher if its symbolic ?
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        args = [e for i, e in enumerate(example_inputs) if i in preserved_arg_indices]
        with fx_traceback.preserve_node_meta():
            gm = make_fx(gm, select_decomp_table())(*args)

    return gm


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    fw_metadata = torch._guards.TracingContext.get().fw_metadata
    params_flat = torch._guards.TracingContext.get().params_flat
    assert fw_metadata is not None and params_flat is not None

    preserved_arg_indices = replace_params_with_constants(
        aot_autograd_gm, params_flat, fw_metadata
    )

    constant_fold(aot_autograd_gm)

    fuse_conv_bn(aot_autograd_gm)
    # now, decomp batch norm if we were unable to fuse it
    aot_autograd_gm = decompose_unfused_batchnorms(
        aot_autograd_gm, example_inputs, preserved_arg_indices
    )
    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    patterns = get_freezing_patterns()

    patterns.apply(aot_autograd_gm.graph)

    # TODO - apply legalization in pattern matcher
    torch.fx.passes.tools_common.legalize_graph(aot_autograd_gm)
    constant_fold(aot_autograd_gm)

    # invalidate nn Modules
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)
    return aot_autograd_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: Optional[str], mod):
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        erased_tensors = [
            e
            for e in pytree.tree_flatten((args, kwargs))[0]
            if isinstance(e, ErasedTensor)
        ]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]

        raise RuntimeError(
            f"Trying to Run Pytorch Eager Module After Dynamo Freezing. "
            "The original parameters have been discarded for memeory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )


@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        if not isinstance(mod, torch.nn.Module):
            continue

        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
            )
        ):
            e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)


@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod):
    for attr_name, tensor in list(
        itertools.chain(
            mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
        )
    ):
        e_t = ErasedTensor(tensor, attr_name, mod)
        if isinstance(tensor, torch.nn.Parameter):
            e_t.requires_grad_(True)
            e_t._is_param = True
        setattr(mod, attr_name, e_t)

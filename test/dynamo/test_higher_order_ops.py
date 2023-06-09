# Owner(s): ["module: dynamo"]
import functools
import unittest
from importlib import import_module

import functorch.experimental.control_flow as control_flow

import torch

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


# Equivalent to backend="eager", but also records graphs that
# we can assert on
class EagerAndRecordGraphs:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        return gm


def count_ops(gm, args, freq, op):
    assert [node.target for node in gm.graph.nodes].count(op) == freq
    return gm


global_var = torch.randn(3)
global_num = 3.14


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


class HigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def _test_wrap_simple(self, func, args, expected_num_wrap_args, expected_opcount=1):
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result
        # - after dynamo capture, the wrap has the expected number of args
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = func(*args)
        result = torch.compile(func, fullgraph=True, backend=cnt)(*args)

        self.assertEqual(result, expected)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, expected_opcount)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertEqual(len(wrap_node.args), expected_num_wrap_args)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 2)

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3)

    def test_capture_constants(self):
        x = torch.randn(3, 3)
        y = 4.0

        def fn(x, y, z):
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            return wrap(fn, x, y, z)

        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

        # Ensure that we recompile here
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            def g(x):
                return wrap(lambda x: x + y, x)

            self._test_wrap_simple(g, (x,), 3)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: x + y, x)

        self._test_wrap_simple(f, (x, y), 3)

    def test_capture_tracked_nested(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        self._test_wrap_simple(f, (x, y), 3)

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, (x, y), 3)

    def test_capture_value_created_in_subgraph(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_map_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=cnt, fullgraph=True)
        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y)

            return control_flow.map(inner, xs, y)

        result = map_f(xs, y)
        self.assertEqual(result, xs + y)

        map_gm = backend.graphs[0]
        name_set = set()
        for name, _ in map_gm.named_modules():
            name_set.add(name)
        self.assertEqual(name_set, {"", "map_body_0.map_body_0", "map_body_0"})

    def test_cond_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            def true_fn(pred2, x, y):
                return x + y

            def false_fn(pred2, x, y):
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)

        cond_gm = backend.graphs[0]
        name_set = set()
        for name, _ in cond_gm.named_modules():
            name_set.add(name)
        self.assertEqual(
            name_set,
            {
                "",
                "cond_true_0",
                "cond_false_0",
                "cond_false_0.cond_false_0",
                "cond_false_0.cond_true_0",
            },
        )

    def test_wrap_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        wrap_gm = backend.graphs[0]
        names = set()
        for mod_name, _ in wrap_gm.named_modules():
            names.add(mod_name)
        self.assertEqual(
            names,
            {
                "",
                "wrap_body_2",
                "wrap_body_2.wrap_body_1",
                "wrap_body_2.wrap_body_1.wrap_body_0",
            },
        )

    def test_capture_global_num(self):
        def f(x):
            return wrap(lambda x: x + global_num, x)

        x = torch.zeros([])
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, (x,), 2)

    def test_capture_global_num_adds_guard(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(lambda x: x + global_num, x)

        global global_num
        x = torch.zeros([])
        result = f(x)
        self.assertEqual(result, x + global_num)

        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        def f(x, y):
            return wrap(lambda x: x + y, x)

        x = torch.zeros([])
        y = 3.14
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, (x, y), 2)

    # TODO: Ideally we would error out if there are any new live side
    # effects (for example, if the body function mutates a global variable).
    # I don't know how to detect this in a robust way, because it conflicts with
    # benign side effects like storing and loading cells that is necessary for
    # capturing variables.
    @unittest.expectedFailure
    def test_side_effect_in_body(self):
        counters.clear()
        backend = EagerAndRecordGraphs()

        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=backend)
        def f(x):
            return wrap(inner, x)

        f(x)
        self.assertEqual(y, x)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"side effects in HigherOrderOperator body": 1},
        )

    def test_fallback_on_graph_break_simple(self):
        # In the future, there should be a per-HigherOrderOperator switch
        # on whether or not to fallback or raise a loud error.
        # For now we just fallback by default.
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(inner, x)

        result = f(x)
        self.assertEqual(result, inner(x))
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to two graphs being compiled
        self.assertEqual(cnt.frame_count, 2)

    def test_fallback_on_graph_break_complicated(self):
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()

        result = f(x)
        self.assertEqual(result, inner(x))
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to four graphs being compiled: clone, sin, sin, clone
        self.assertEqual(cnt.frame_count, 4)

    def test_modules(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        # 3 args - 1 for input, and other 2 for the weight and bias
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the linear bias and weight are getattr in the outer graph
        self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        linear_node = find_first_node(body_function, torch._C._nn.linear)
        self.assertTrue(linear_node is not None)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_flat_list_output(self):
        def f(x):
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 2, expected_opcount=3)

    def test_fallback_on_python_primitives_output(self):
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [1, torch.sin(x), 2.0])
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator body's output must consist of tensors only": 1},
        )

    def test_fallback_on_nested_tuple_output(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: ((x.sin(), x.cos()),), x)

        x = torch.randn(2, 3)
        result = f(x)

        self.assertEqual(result, ((x.sin(), x.cos()),))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator body's output must consist of tensors only": 1},
        )

    def test_fallback_on_output_with_dict(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: [{"a": -x}], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [{"a": -x}])
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator body's output must consist of tensors only": 1},
        )

    def test_access_module_attr(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the linear bias and weight are getattr in the outer graph
        self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_make_closure(self):
        def f(x, y):
            def g(x):
                return x + y

            return g(x)

        def h(x, y):
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(h, (x, y), 3)

    def test_capture_numpy_number(self):
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        self._test_wrap_simple(f, (x,), 3)

    def test_freevars_as_inputs_to_wrap(self):
        y = torch.randn(3)

        def f(x):
            return wrap(lambda x, y: x + y, x, y)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3)

    def test_lift_tensor_constant(self):
        def f(x):
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3, expected_opcount=2)

    def test_nested_wrap(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        mod = MockModule()

        # Two levels of wrap ops
        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            return wrap(gn, x)

        self._test_wrap_simple(fn, (torch.randn(10, 10),), 4, expected_opcount=1)

    def test_hooks(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        forward_handles = {}
        activations = dict()

        def save_activations(mod, inp, out):
            activations[name] = inp

        for name, module in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return wrap(lambda x: model(x), x)

        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((10, 10))
            pred = fn(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(activations.keys() == forward_handles.keys())


class ActivationCheckpointingTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.cos(torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y))

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 2)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda()
    def test_tags_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_multiple_checkpoints(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    @requires_cuda()
    def test_tags_decomps(self):
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        def gn(x, y):
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            # x = torch.utils.checkpoint.checkpoint(gn, x, y)
            return x

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x)

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

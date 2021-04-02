# This example is provided only for explanatory and educational purposes.
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import Proxy
from typing import Tuple, Any, Optional

import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

# Vmap
# ---------------
# `vmap` (short for vectorizing map) is a function transformation that takes in
# a model that operates on single examples and returns a model that operates on
# multiple examples. For example, if our model `M` originally takes in tensors
# of shape (H, W) and returns a scalar, then `vmap(M)` should take in tensors
# of shape `(B, H, W)` and return a vector. This procedure is also often called
# "batching" a model.
#
# How is this feat accomplished? One observation is that to "batch" a model, it
# suffices to batch each individual operation. In other words, given an
# operation that works on the current shape, how do we make it work with an
# additional batch dimension? This leads us to batching rules.
#
# Batching Rules
# ---------------
# A batching rule for a function `f` takes in the function `f` (that operates
# on unbatched values), a batched argument `x`, and performs the necessary
# transformations to apply `f` to `x`.
#
# One simple example is `torch.movedim(x, from_dim, to_dim)`, which moves a
# dimension from `from_dim` to `to_dim`. For example, if `x.shape = (1,2,3,4)`,
# then torch.movedim(x, 0, 2) would result in a shape of `(2,3,1,4)`.
#
# However, let's say that we introduce a batch dimension - `x.shape =
# (B,1,2,3,4)`. Now, we can't simply execute the same `torch.movedim(x,0,2)`,
# as there is an extra batch dimension in the front. Instead, we must execute
# `torch.movedim(x,1,3)`. This procedure (and some other stuff to make sure the
# batch dimension is always at the front) is what's done in
# `movedim_batching_rule`.
#
# There is one final thing to note about these batching rules - they're almost
# entirely written in normal PyTorch, with the exception of `bdim` attribute
# that's needed for tracking the batch dimension. That is because in order to
# use these batching rules, we will be tracing them by passing in `Proxy`
# objects that will track the operations performed on them and append them to
# the graph.

def move_bdim_to_front(x, result_ndim=None):
    """
    Returns a tensor with a batch dimension at the front. If a batch
    dimension already exists, move it. Otherwise, create a new batch
    dimension at the front. If `result_ndim` is not None, ensure that the
    resulting tensor has rank equal to `result_ndim`.
    """
    x_dim = len(x.shape)
    x_bdim = x.bdim
    if x_bdim is None:
        x = torch.unsqueeze(x, 0)
    else:
        x = torch.movedim(x, x_bdim, 0)
    if result_ndim is None:
        return x
    diff = result_ndim - x_dim - (x_bdim is None)
    for _ in range(diff):
        x = torch.unsqueeze(x, 1)
    return x

def movedim_batching_rule(x, from_dim, to_dim):
    x = move_bdim_to_front(x)
    return torch.movedim(x, from_dim + 1, to_dim + 1), 0

batching_rules = {}
def gen_binary_op_batching_rule(op):
    def binary_op_batching_rule(a, b):
        a_ndim = len(a.shape)
        b_ndim = len(b.shape)
        result_ndim = max(a_ndim, b_ndim)
        a = move_bdim_to_front(a, result_ndim)
        b = move_bdim_to_front(b, result_ndim)
        res = op(a, b)
        return res, 0
    return binary_op_batching_rule

def unsqueeze_batching_rule(x, dim):
    x = move_bdim_to_front(x)
    if dim >= 0:
        return torch.unsqueeze(x, dim + 1), 0
    else:
        return torch.unsqueeze(x, dim), 0


batching_rules[torch.mul] = gen_binary_op_batching_rule(torch.mul)
batching_rules[torch.unsqueeze] = unsqueeze_batching_rule
batching_rules[torch.movedim] = movedim_batching_rule


# In order to apply a batching rule, we will simply pass in `Proxy` objects as
# inputs to the functions. As the batching rules need some extra information
# such as the batch dimension and shape, we will do some bookkeeping here.
def gen_batching_rule_function(target, *args):
    def lift_shape(i):
        res = Proxy(i)
        res.shape = i.shape
        res.bdim = i.bdim
        return res
    proxy_args = [lift_shape(i) if isinstance(i, fx.Node) else i for i in args]
    out, bdim = batching_rules[target](*proxy_args)
    out_node = out.node
    out_node.bdim = bdim
    return out_node

def vmap(model: torch.nn.Module, in_axes: Tuple[Optional[int], ...], example_args: Tuple[Any, ...]) -> torch.nn.Module:
    """vmap
    Given a model with inputs, vmap will return a function that works on
    batched versions of those inputs. Which inputs will be batched is
    determined by in_axes. In addition, as vmap requires shape (actually
    rank) information, we will pass in example_args (example inputs for the
    original module).
    """
    in_axes = iter(in_axes)
    fx_model = fx.symbolic_trace(model)
    # Here we run a shape propagation pass in order to annotate the graph with shape information.
    ShapeProp(fx_model).propagate(*example_args)
    # As vmap rewrites the whole graph, it's easiest to create an entirely new
    # graph and append to that.
    new_graph: fx.Graph = fx.Graph()

    # We will create an environment to map the new nodes created to the
    # corresponding old nodes.
    def lookup_env(l):
        return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
    env = {}
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # If the node is an input placeholder, we simply copy it over and
            # annotate it with the batch dimension from `in_axes`.
            new_node = new_graph.placeholder(node.name)
            new_node.bdim = next(in_axes)
            new_node.meta = node.meta
            env[node.name] = new_node
        elif node.op == 'output':
            new_graph.output(env[node.args[0].name])
        elif node.op == 'call_function':
            new_args = lookup_env(node.args)
            # If any of the inputs to the function has a new batch dimension,
            # we will need to use our batching rules. Otherwise, we will simply
            # copy the node over.
            if any([x.bdim is not None for x in new_args if isinstance(x, fx.Node)]):
                new_node = gen_batching_rule_function(node.target, *new_args)
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                new_node.bdim = None
            new_node.meta = node.meta
            env[node.name] = new_node
        else:
            raise RuntimeError("Not yet implemented")


    res = fx.GraphModule(fx_model, new_graph)
    print(res.code)
    res.graph.lint()
    return res

vjp_map = {}

def add_vjp(g, a, b):
    if isinstance(a, fx.Proxy) and isinstance(b, fx.Proxy):
        assert(a.shape == b.shape)
    return g, g

def sub_vjp(g, a, b):
    if isinstance(a, fx.Proxy) and isinstance(b, fx.Proxy):
        assert(a.shape == b.shape)
    return g, -g

def mul_vjp(g, a, b):
    if isinstance(a, fx.Proxy) and isinstance(b, fx.Proxy):
        assert(a.shape == b.shape)
    return g * b, g * a

def div_vjp(g, a, b):
    if isinstance(a, fx.Proxy) and isinstance(b, fx.Proxy):
        assert(a.shape == b.shape)
    return g *(1/b), g * (-a / (b**2.0))

def pow_vjp(g, a, b):
    if isinstance(b, fx.Proxy):
        raise RuntimeError("nyi")
    return g * b*a*(b-1), None

def sum_vjp(g, x):
    return g.expand(x.shape)

vjp_map[torch.add] = add_vjp
vjp_map[torch.sub] = sub_vjp
vjp_map[torch.mul] = mul_vjp
vjp_map[torch.div] = div_vjp
vjp_map[torch.sum] = sum_vjp
vjp_map[torch.pow] = pow_vjp

def grad(model: torch.nn.Module, example_inps: Tuple[Any, ...], get_value=True) -> torch.nn.Module:
    fx_model = fx.symbolic_trace(model)
    ShapeProp(fx_model).propagate(*example_inps)
    # graph and append to that.
    val_map = {}
    new_graph: fx.Graph = fx.Graph()
    orig_output = new_graph.graph_copy(fx_model.graph, val_map)
    def shape_proxy(node):
        proxy = fx.Proxy(val_map[node])
        proxy.shape = node.meta['shape']
        proxy.dim = lambda : len(proxy.shape)
        return proxy
    inputs = []
    ones = new_graph.create_node('call_function', torch.ones, ([],))

    for node in reversed(fx_model.graph.nodes):
        if node.op == 'output':
            assert(len(node.args) == 1)
            val_map[node.args[0]].grad = [fx.Proxy(ones)]
        elif node.op == 'placeholder':
            inputs.append(sum(val_map[node].grad).node)
        elif node.op == 'call_function':
            g = sum(val_map[node].grad)
            new_args = [shape_proxy(i) if isinstance(i, fx.Node) else i for i in node.args]
            if node.target not in vjp_map:
                raise RuntimeError("vjp not yet implemented")
            new_grads = vjp_map[node.target](g, *new_args)
            if not isinstance(new_grads, tuple):
                new_grads = (new_grads,)
            for new_g, arg in zip(new_grads, new_args):
                if isinstance(arg, fx.Proxy):
                    if not hasattr(arg.node, 'grad'):
                        arg.node.grad = []
                    arg.node.grad.append(new_g)
        elif node.op == 'call_method':
            raise RuntimeError("doesn't support methods since i'm lazy")

    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = inputs[::-1]
    if get_value:
        new_graph.output((orig_output, inputs))
    else:
        new_graph.output(inputs)
    res = fx.GraphModule(fx_model, new_graph)
    res.graph.lint()
    return res



if __name__ == '__main__':
    from nnc_compile import decompose
    def f(x):
        return 0.5 * torch.sum(x)
    example_inps = (torch.randn(5),)
    f = decompose(f, example_inputs=example_inps)
    f_g = grad(f, example_inps=example_inps, get_value=True)

    out =  torch.autograd.functional.vjp(f, example_inps)
    print(out)
    print(f_g(*example_inps))

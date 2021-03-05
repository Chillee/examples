#, This example is provided only for explanatory and educational purposes. The
# underlying APIs may change and this tutorial may break.

# Compiling FX models to NNC (Neural Network Compiler)
######################################################
# The goal of this file is to demonstrate an end to end example of using FX to
# lower a PyTorch model to a backend codegen compiler. In this example, we will
# be using NNC
# (https://github.com/pytorch/pytorch/blob/master/test/cpp/tensorexpr/tutorial.cpp).
# If you're unfamiliar with NNC, the general design is strongly inspired by TVM
# and Halide.
#
# To do so, this example contains two FX transformations.
# The first one is a decomposition pass that normalizes and decomposes PyTorch
# operations (such as addmm). Using a pass like this allows us to reduce the
# number of lowerings we need to write. Instead of needing to specifically
# write a lowering for addmm, we can decompose addmm and lower its constituent
# operations.
# The second one is the actual lowering pass itself. In this case, we will need
# to convert each PyTorch operation we encounter into the corresponding NNC
# `TensorExpr`.
#
# These two passes, `decompose` and `nnc_compile`, are fairly similar.
# In both cases, we re-interpret each operation in the FX graph to construct an
# entirely new representation. In the decomposition pass, we either copy the
# operation as-is into the new graph, or we use `Proxy` objects to decompose
# the operation. This is an extension of the example presented here:
# https://pytorch.org/docs/master/fx.html#proxy-retracing
#
# In the lowering pass, a similar principle applies. However, instead of using
# `Proxy` objects to rewrite our op in other PyTorch ops, we do the translation
# ourselves. In addition, since this is not a source-to-source transformation,
# we return a somewhat hacky function that passes in the module attributes to
# the NNC callable.
#
# Results
######################################
# Using NNC (which compiles directly to LLVM), we can compile a fairly small
# PyTorch model and compare performnance between NNC, PyTorch Eager, and Static
# Runtime. These are my resuls on an Intel i7-8750H CPU.
#
# NNC time:  0.0066373348236083984
# PyTorch time 0.025979042053222656
# Static Runtime time 0.011004209518432617
#
# As we can see, NNC is nearly 2x faster than static runtime and more than 4x
# faster than PyTorch. This is not surprising, as we are dealing with extremely
# small tensors where framework overhead is a significant factor.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C._te as te
import torch.fx as fx
from torch.fx import map_arg
from torch.fx.passes.shape_prop import ShapeProp
import operator
scope = te.KernelScope()

# Decomposition Pass

def binary_mapping(op):
    def f(a, b):
        return op(a, b)
    return f

decomposition_rules = {}
binary_decompositions = [
    (operator.matmul, torch.mm),
    (operator.add, torch.add),
    (operator.mul, torch.mul),
    (operator.sub, torch.sub),
    (operator.truediv, torch.div),
    (operator.eq, torch.eq),
    (operator.gt, torch.gt),
    (operator.ge, torch.ge),
    (operator.lt, torch.lt),
    (operator.le, torch.le),
    (operator.ne, torch.ne),
    (operator.and_, torch.bitwise_and),
    (operator.pow, torch.pow)
]
for old, new in binary_decompositions:
    decomposition_rules[old] = binary_mapping(new)

def addmm_decompose(input, mat1, mat2, beta=1, alpha=1, out=None):
    assert(out is None)
    return beta*input + alpha*(torch.mm(mat1, mat2))

def matmul_decompose(A, B, out=None):
    assert(out is None)
    if A.dim() == 1 and B.dim() == 1:
        return torch.dot(A, B)
    if A.dim() == 2 and B.dim() == 2:
        return torch.mm(A, B)
    if A.dim() == 2 and B.dim() == 1:
        return torch.mv(A, B)
    else:
        raise RuntimeError("nyi")

decomposition_rules[torch.addmm] = addmm_decompose
decomposition_rules[operator.neg] = lambda x: 0 - x
decomposition_rules[torch.matmul] = matmul_decompose


def method_torch_decompose(method):
    def f(self, *args):
        return getattr(torch, method)(self, *args)
    return f

tensor_methods_simple = ['add', 'sub', 'mul', 'div', 'matmul']

for meth in tensor_methods_simple:
    decomposition_rules[(torch.Tensor, meth)] = method_torch_decompose(meth)

def method_reshape_decompose(self, *shape):
    return torch.reshape(self, shape)

decomposition_rules[(torch.Tensor, 'reshape')] = method_reshape_decompose

def method_t_decompose(self):
    assert(self.dim() <= 2)
    if self.dim() < 2: return self
    else: return torch.transpose(self, 1, 0)
decomposition_rules[(torch.Tensor, 't')] = method_t_decompose

def module_conv2d_decompose(module, x):
    return F.conv2d(x, module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

def module_relu6_decompose(module, x):
    return torch.clamp(x, 0.0, 6.0)

def module_dropout_decompose(module, x):
    if module.training == False:
        return x
    else:
        return F.dropout(x, module.p, module.training)

def module_linear_decompose(module, x):
    weight, bias = module.weight, module.bias
    if x.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, x, weight.t())
    else:
        output = x.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret

decomposition_rules[nn.Conv2d] = module_conv2d_decompose
decomposition_rules[nn.ReLU6] = module_relu6_decompose
decomposition_rules[nn.Dropout] = module_dropout_decompose
decomposition_rules[nn.Linear] = module_linear_decompose

def shape_proxy(node):
    proxy = fx.Proxy(node)
    if hasattr(node, 'shape'):
        proxy.shape = node.shape
        proxy.dim = lambda : len(proxy.shape)
    return proxy

def fetch_attr(module, target : str):
    target_atoms = target.split('.')
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

def remove_shapes(model, example_inputs):
    model = fx.symbolic_trace(model)
    ShapeProp(model).propagate(*example_inputs)
    new_graph = fx.Graph()
    env = {}
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target == torch.reshape:
            proxy_args = map_arg(node.args, lambda n: shape_proxy(env[n.name]))
            proxy_kwargs = map_arg(node.kwargs, lambda n: shape_proxy(env[n.name]))
            new_node = torch.reshape(proxy_args[0], node.shape).node
            new_node.shape = node.shape
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            if hasattr(node, 'shape'):
                new_node.shape = node.shape
            env[node.name] = new_node
    new_graph.lint()
    model = fx.GraphModule(model, new_graph)
    return model



def decompose(model: torch.nn.Module, example_inputs) -> torch.nn.Module:
    """
    decompose(model, example_inputs) takes in a model, decomposes any of the functions in `decomposition_rules` to its constituent operations, and returns a `nn.Module` without any of the operations with decomposition rules.
    """
    # Run it multiple times so we converge to a fixed point.
    for _ in range(5):
        model = fx.symbolic_trace(model)
        ShapeProp(model).propagate(*example_inputs)
        new_graph = fx.Graph()
        env = {}
        for node in model.graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # If the current function is in `decomposition_rules`, we use
                # `Proxy` objects to decompose the operations using the
                # decomposition rule. See
                # https://pytorch.org/docs/master/fx.html#proxy-retracing for
                # more details.
                proxy_args = map_arg(node.args, lambda n: shape_proxy(env[n.name]))
                proxy_kwargs = map_arg(node.kwargs, lambda n: shape_proxy(env[n.name]))
                new_node = decomposition_rules[node.target](*proxy_args, **proxy_kwargs).node
                new_node.shape = node.shape
                env[node.name] = new_node
            elif node.op == 'call_method' and (torch.Tensor, node.target) in decomposition_rules:
                proxy_args = map_arg(node.args, lambda n: shape_proxy(env[n.name]))
                proxy_kwargs = map_arg(node.kwargs, lambda n: shape_proxy(env[n.name]))
                new_node = decomposition_rules[(torch.Tensor, node.target)](*proxy_args, **proxy_kwargs).node
                new_node.shape = node.shape
                env[node.name] = new_node
            elif node.op == 'call_module' and type(fetch_attr(model, node.target)) in decomposition_rules:
                proxy_args = map_arg(node.args, lambda n: shape_proxy(env[n.name]))
                proxy_kwargs = map_arg(node.kwargs, lambda n: shape_proxy(env[n.name]))
                module = fetch_attr(model, node.target)
                unset_keys = []
                for key in module.state_dict():
                    module.__dict__[key] = fx.Proxy(new_graph.get_attr(node.target+'.' + key))
                    unset_keys.append(key)
                new_node = decomposition_rules[type(module)](module, *proxy_args, **proxy_kwargs).node
                for key in unset_keys:
                    del module.__dict__[key]
                new_node.shape = node.shape
                env[node.name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                if hasattr(node, 'shape'):
                    new_node.shape = node.shape
                env[node.name] = new_node
        new_graph.lint()
        model = fx.GraphModule(model, new_graph)
    model = remove_shapes(model, example_inputs)
    return model

# NNC Lowering Pass

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

def get_dim_args(dims):
    dim_args = []
    for dim in dims:
        dim_args.append(te.DimArg(te.ExprHandle.int(dim), 'i' + str(len(dim_args))))
    return dim_args

def get_te_shapes(shape):
    return [te.ExprHandle.int(i) for i in shape]

def to_expr(x):
    if isinstance(x, int):
        return te.ExprHandle.int(x)
    elif isinstance(x, float):
        return te.ExprHandle.float(x)
    else:
        raise RuntimeError(f"type {type(x)} not supported")

def get_nnc_type(dtype):
    if dtype == torch.float:
        return te.Dtype.Float
    elif dtype == torch.long:
        return te.Dtype.Long
    else:
        raise RuntimeError("nyi")


lowering_functions = {}

def wrap_compute(f):
    def fn_lower(name, out_shape, inp_shapes, args):
        if len(out_shape) == 0:
            out_shape = [1]
        X = te.Compute(name, get_dim_args(out_shape), f(inp_shapes, args))
        return X
    return fn_lower

def gen_unary_nnc(op):
    def gen_op_nnc(inp_shapes, args):
        def f(*idxs):
            return op(args[0].load(idxs))
        return f
    return gen_op_nnc

unary_lowerings = [
    (torch.sin, lambda x: x.sin()),
    (torch.cos, lambda x: x.cos()),
    (torch.tan, lambda x: x.tan()),
    (torch.asin, lambda x: x.asin()),
    (torch.acos, lambda x: x.acos()),
    (torch.atan, lambda x: x.atan()),
    (torch.sinh, lambda x: x.sinh()),
    (torch.cosh, lambda x: x.cosh()),
    (torch.tanh, lambda x: x.tanh()),
    (torch.sigmoid, lambda x: x.sigmoid()),
    (torch.exp, lambda x: x.exp()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.abs, lambda x: x.abs()),
    (torch.log, lambda x: x.log()),
    (torch.log2, lambda x: x.log2()),
    (torch.log10, lambda x: x.log10()),
    (torch.log1p, lambda x: x.log1p()),
    (torch.erf, lambda x: x.erf()),
    (torch.erfc, lambda x: x.erfc()),
    (torch.sqrt, lambda x: x.sqrt()),
    (torch.rsqrt, lambda x: x.rsqrt()),
    (torch.ceil, lambda x: x.ceil()),
    (torch.floor, lambda x: x.floor()),
    (torch.round, lambda x: x.round()),
    (torch.trunc, lambda x: x.trunc()),
    (torch.lgamma, lambda x: x.lgamma()),
]

for torch_op, nnc_fn in unary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_unary_nnc(nnc_fn))

def gen_binary_nnc(op):
    def is_nnc_obj(x):
        return isinstance(x, te.Placeholder) or isinstance(x, te.Tensor) or isinstance(x, te.BufHandle)
    def gen_op_nnc(inp_shapes, args):
        if is_nnc_obj(args[0]) and is_nnc_obj(args[1]):
            A_shape, A_dtype = inp_shapes[0]
            B_shape, B_dtype = inp_shapes[1]
            A, B = args

            def index_or_broadcast(shape, *args):
                out = []
                for idx, arg in enumerate(args):
                    if idx >= len(shape): continue
                    if shape[idx] == 1:
                        out.append(to_expr(0))
                    else:
                        out.append(arg)
                return out

            def f(*idxs):
                return op(A.load(index_or_broadcast(A_shape, *idxs)), B.load(index_or_broadcast(B_shape, *idxs)))
            return f
        else:
            if is_nnc_obj(args[0]):
                def f(*idxs):
                    return op(args[0].load(idxs), to_expr(args[1]))
                return f
            else:
                def f(*idxs):
                    return op(to_expr(args[0]), args[1].load(idxs))
                return f

    return gen_op_nnc


binary_lowerings = [
(torch.add,lambda a, b: a+b),
(torch.mul,lambda a, b: a*b),
(torch.sub,lambda a, b: a-b),
(torch.div,lambda a, b: a/b),
(torch.eq,lambda a, b: a==b),
(torch.gt,lambda a, b: a>b),
(torch.lt,lambda a, b: a<b),
(torch.ge,lambda a, b: a>=b),
(torch.le,lambda a, b: a<=b),
(torch.pow, lambda a, b: te.pow(a, b)),
]
for torch_op, nnc_fn in binary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_binary_nnc(nnc_fn))

def clamp_lower(inp_shapes, args):
    def f(*idxs):
        val = args[0].load(idxs)
        # return val
        # return te.compareSelect(val, to_expr(args[1]), val, to_expr(args[1]), te.CompareSelectOperation.GE)
        return te.ifThenElse(val < to_expr(args[1]), to_expr(args[1]),
                            te.ifThenElse(val > to_expr(args[2]), to_expr(args[2]), val))
    return f

lowering_functions[torch.clamp] = wrap_compute(clamp_lower)

def transpose_lower(name, out_shape, inp_shapes, args):
    idx_1, idx_2 = args[1], args[2]
    def transpose(shape):
        shape[idx_1], shape[idx_2] = shape[idx_2], shape[idx_1]
        return shape
    def f(*idxs):
        idxs = transpose(list(idxs))
        return args[0].load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def prod(x, start=1):
    t = start
    for i in x: t *= i
    return t

def flatten_lower(name, out_shape, inp_shapes, args):
    A, start_dim, end_dim = args
    shape = list(inp_shapes[0][0])
    if end_dim < 0:
        end_dim = len(shape) + end_dim
    flattened_region = shape[start_dim:end_dim+1]
    def get_orig_idxs(i):
        idxs = []
        total = prod(flattened_region)
        for dim in flattened_region:
            total //= dim
            idxs.append(i / to_expr(total))
            i = i % to_expr(total)
        return idxs
    def f(*idxs):
        idxs = list(idxs)
        idxs = idxs[:start_dim] + get_orig_idxs(idxs[start_dim]) + idxs[start_dim+1:]
        return A.load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def cat_lower(name, out_shape, inp_shapes, args):
    tensors = args[0]
    dim = args[1]
    lengths = [i[0][dim] for i in inp_shapes[0]]
    def f(*idxs):
        idxs = list(idxs)
        sm = lengths[0]
        load = tensors[0].load(idxs)
        for length, tensor in list(zip(lengths, tensors))[1:]:
            new_idxs = idxs[:]
            new_idxs[dim] -= to_expr(sm)
            load = te.ifThenElse(idxs[dim] < to_expr(sm), load, tensor.load(new_idxs))
        return load
    return te.Compute(name, get_dim_args(out_shape), f)

lowering_functions[torch.transpose] = transpose_lower
lowering_functions[torch.flatten] = flatten_lower
lowering_functions[torch.cat] = cat_lower

def bmm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    B, N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][2]

    def f(b, n, p, m):
        return M1.load([b, n, m]) * M2.load([b, m, p])
    mm = te.Compute('mm', get_dim_args([B,N,P,M]), f)
    out_tensor = te.Reduce(name, get_dim_args([B, N, P]), te.Sum(), mm, get_dim_args([M]))
    return out_tensor.buf(), [mm.stmt(), out_tensor.stmt()]


def mm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][1]

    # def f(n, p, m):
    #     return M1.load([n, m]) * M2.load([m, p])
    # mm = te.Compute('mm', get_dim_args([N,P,M]), f)
    # out = te.Reduce(name, get_dim_args([N, P]), te.Sum(), mm, get_dim_args([M]))
    # return out.buf(), [mm.stmt(), out.stmt()]
    C = torch._C._te.BufHandle('C', get_te_shapes([N, P]), get_nnc_type(torch.float))
    s = torch._C._te.ExternalCall(C, "nnc_aten_matmul", [M1, M2], [])
    return C, [s]

def mv_lower(name, out_shape, inp_shapes, args):
    A = args[0]
    B = args[1]
    N, M = inp_shapes[0][0]

    # def f(n, m):
    #     return A.load([n, m]) * B.load([m])
    # mm = te.Compute('mm', get_dim_args([N,M]), f)
    # out = te.Reduce(name, get_dim_args([N]), te.Sum(), mm, get_dim_args([M]))
    # return out.buf(), [mm.stmt(), out.stmt()]
    C = torch._C._te.BufHandle('C', get_te_shapes([N, M]), get_nnc_type(torch.float))
    s = torch._C._te.ExternalCall(C, "nnc_aten_mv", [A, B], [])
    return C, [s]

def dot_lower(name, out_shape, inp_shapes, args):
    A, B = args
    N = inp_shapes[0][0][0]
    def f(n):
        return A.load([n]) * B.load([n])
    res = te.Compute('mul', get_dim_args([N]), f)
    out = te.Reduce(name, get_dim_args([]), te.Sum(), res, get_dim_args([N]))
    return out.buf(), [res.stmt(), out.stmt()]

lowering_functions[torch.bmm] = bmm_lower
lowering_functions[torch.mm] = mm_lower
lowering_functions[torch.mv] = mv_lower
lowering_functions[torch.dot] = dot_lower

def sum_lower(name, out_shape, inp_shapes, args):
    assert(len(args) == 1)
    A = args[0]
    return te.Reduce(name, get_dim_args([]), te.Sum(), A, get_dim_args(inp_shapes[0][0]))

lowering_functions[torch.sum] = sum_lower

def conv2d_lower(name, out_shape, inp_shapes, args):
    X, W, B, stride, padding, dilation, groups = args
    out = te.BufHandle('out', get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))

    conv_spec = list(stride)+list(padding) + list(dilation) + [groups]
    conv_spec = [to_expr(i) for i in conv_spec]
    s = te.ExternalCall(out, "nnc_aten_conv2d", [X, W, B], conv_spec)
    return out, [s]

def adaptive_avg_pool2d_lower(name, out_shape, inp_shapes, args):
    X, output_size = args
    out = te.BufHandle('out', get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    output_size = [to_expr(i) for i in output_size]
    s = te.ExternalCall(out, "nnc_aten_adaptive_avg_pool2d", [X], output_size)
    return out, [s]

def encode_idxs(shape, idxs):
    assert(len(shape) == len(idxs))
    cur = 1
    out = to_expr(0)
    for dim, idx in reversed(list(zip(shape, idxs))):
        out += to_expr(cur) * idx
        cur *= dim
    return out

def reshape_lower(name, out_shape, inp_shapes, args):
    X, shape = args
    start_shape = list(inp_shapes[0][0])
    end_shape = out_shape
    def get_orig_idxs(idxs):
        absolute_new = encode_idxs(end_shape, idxs)
        new_idxs = []
        total_old = prod(start_shape)
        for dim in start_shape:
            total_old //= dim
            new_idxs.append(absolute_new / to_expr(total_old))
            absolute_new %= to_expr(total_old)
        return new_idxs

    def f(*idxs):
        idxs = list(idxs)
        orig_idxs = get_orig_idxs(idxs)
        return X.load(orig_idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

lowering_functions[F.conv2d] = conv2d_lower
lowering_functions[F.adaptive_avg_pool2d] = adaptive_avg_pool2d_lower
lowering_functions[torch.reshape] = reshape_lower


def lower_function(node, op, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: (arg.shape, arg.dtype) if isinstance(arg, fx.Node) and hasattr(arg, 'shape') else None)
    out = lowering_functions[op](node.name, node.shape, inp_shapes, nnc_args)
    if isinstance(out, te.Tensor):
        return out.buf(), [out.stmt()]
    else:
        return out[0], out[1]
        print(out)
        raise RuntimeError("nyi")

def nnc_compile(model: torch.nn.Module, example_inputs) -> torch.nn.Module:
    """
    nnc_compile(model, example_inputs) returns a function with the same args
    as `model.forward`, with an extra argument corresponding to where the
    output is stored. This function takes the inputs (which must be PyTorch
    tensors with the same shapes as example_inputs), and passes them to an
    NNC executor.
    """
    fx_model = fx.symbolic_trace(model)
    ShapeProp(fx_model).propagate(*example_inputs)

    # This env maps from nodes to `te.ExprHandle`, which represent the output
    # of an NNC computation.
    env = {}


    def get_te_type(node):
        return get_nnc_type(node.dtype)

    def gen_compute(args):
        te_args = [env[arg.name] for arg in args]

    def lookup_env(l):
        res = fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
        return res

    def fetch_attr(target : str):
        target_atoms = target.split('.')
        attr_itr = fx_model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    outs = None
    inputs = []
    module_attrs = []
    compute_stmts = []
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # We simply map the input placeholder to a `te.Placeholder`, which
            # also represents an input to the NNC computation.
            shapes = get_te_shapes(node.shape)
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.buf()
            inputs.append(placeholder)
        elif node.op == 'call_function':
            # This does the bulk of the work - we call `lower_function`, which
            # returns a `te.ExprHandle` (the output of a NNC computation), and
            # put it in our environment.
            if hasattr(node, 'shape'):
                # todo: fix kwargs handling
                if node.kwargs:
                    raise RuntimeError("kwargs nyi")
                buf, stmt = lower_function(node, node.target, lookup_env(node.args), node.args)
                # if isinstance(stmt, list)
                compute_stmts.extend(stmt)
                env[node.name] = buf
            elif node.target == getattr or node.target == operator.getitem:
                # todo: handle non-tensor computations correctly
                continue
        elif node.op == 'output':
            args = node.args
            if not isinstance(args, tuple):
                args = (args,)
            if isinstance(args[0], tuple):
                args = args[0]
            te_args = lookup_env(args)
            outs = (list(te_args), [i.shape for i in args])
        elif node.op == 'get_attr':
            # As NNC doesn't have any concept of state, we pull out the module
            # attributes and pass them in as inputs to NNC.
            module_attrs.append(node)
            shapes = get_te_shapes(node.shape)
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.buf()
        else:
            print(node.op, node.target)
            raise RuntimeError("not yet implemented")


    loopnest = te.LoopNest(te.Stmt(compute_stmts), outs[0])
    # loopnest.inline_intermediate_bufs(True)
    loopnest.simplify()
    # print(loopnest)
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + outs[0]])
    alloc_results = [torch.empty(i) for i in outs[1]]
    def f(*inps, out_tensors=None):
        if out_tensors is None:
            results = alloc_results
        else:
            results = out_tensors
        if module_attrs:
            module_stuff = [fetch_attr(i.target).data for i in module_attrs]
        else:
            module_stuff = []
        cg.call(module_stuff + list(inps) + results)
        if out_tensors is None:
            if len(results) == 1:
                return results[0]
            return results
    return f


################################
# Example usage and Benchmarking
################################

if __name__ == '__main__':
    class DeepAndWide(torch.nn.Module):
        def __init__(self, num_features=50):
            super(DeepAndWide, self).__init__()
            self.mu = torch.nn.Parameter(torch.randn(1, num_features))
            self.sigma = torch.nn.Parameter(torch.randn(1, num_features))
            self.fc_w = torch.nn.Parameter(torch.randn(1, num_features + 1))
            self.fc_b = torch.nn.Parameter(torch.randn(1))

        def forward(self, ad_emb_packed, user_emb, wide):
            wide_offset = wide + self.mu
            wide_normalized = wide_offset * self.sigma
            wide_preproc = torch.clamp(wide_normalized, 0., 10.)
            user_emb_t = torch.transpose(user_emb, 1, 2)
            dp_unflatten = torch.bmm(ad_emb_packed, user_emb_t)
            dp = torch.flatten(dp_unflatten, 1, -1)
            inp = torch.cat([dp, wide_preproc], 1)
            t1 = torch.transpose(self.fc_w, 1, 0)
            fc1 = torch.addmm(self.fc_b, inp, t1)
            return fc1

    with torch.no_grad():
        num_features = 50
        mod = DeepAndWide(num_features)
        # Phabricate sample inputs
        batch_size = 1
        embedding_size = 32
        ad_emb_packed = torch.randn(batch_size, 1, embedding_size)
        user_emb = torch.randn(batch_size, 1, embedding_size)
        wide = torch.randn(batch_size, num_features)
        inps = (ad_emb_packed, user_emb, wide)

        output_shape = mod(ad_emb_packed, user_emb, wide).shape
        out = torch.empty(output_shape)

        mod = decompose(mod, inps)
        cg = nnc_compile(mod, inps)

        iters = 100

        for _ in range(10):
            cg(ad_emb_packed, user_emb,wide, out_tensors=[out])
        begin = time.time()
        for _ in range(iters):
            cg(ad_emb_packed, user_emb,wide, out_tensors=[out])

        print("NNC time: ", time.time()-begin)

        mod_jit = torch.jit.script(DeepAndWide(num_features))
        for _ in range(10):
            mod_jit(ad_emb_packed, user_emb,wide)
        begin = time.time()
        for _ in range(iters):
            mod_jit(ad_emb_packed, user_emb,wide)
        print("PyTorch time", time.time()-begin)

        static_runtime = torch._C._jit_to_static_runtime(mod_jit._c)
        for _ in range(10):
            static_runtime.run([ad_emb_packed, user_emb,wide])
        begin = time.time()
        for _ in range(iters):
            static_runtime.run([ad_emb_packed, user_emb,wide])
        print("Static Runtime time", time.time()-begin)

        print("Sums:", out.sum(), mod(*inps).sum())

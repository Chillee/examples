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

def truncate(model, k):
    model = fx.symbolic_trace(model)
    new_graph= fx.Graph()
    env = {}

    cnt = 0
    for node in list(model.graph.nodes):
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
        cnt += 1
        if cnt == k:
            new_graph.output(env[node.name])
            break

    return fx.GraphModule(model, new_graph)

# NNC Lowering Pass
def remove_args(model: torch.nn.Module):
    fx_model = fx.symbolic_trace(model)
    for node in fx_model.graph.nodes:
        if len(node.users) == 0 and node.op != 'output':
            fx_model.graph.erase_node(node)
    fx_model.recompile()
    return fx_model

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
    elif dtype == torch.float64:
        return te.Dtype.Double
    else:
        raise RuntimeError("nyi")


lowering_functions = { }
def index_or_broadcast(shape, *args):
    out = []
    for idx, arg in enumerate(args):
        if idx >= len(shape): continue
        if shape[idx] == 1:
            out.append(to_expr(0))
        else:
            out.append(arg)
    return out

def ones_like_lower(name, out_shape, inp_shapes, args):
    def f(*idxs):
        return to_expr(1.0)
    res = te.Compute(name, get_dim_args(out_shape), f)
    return res

# def select_lower(name, out_shape, inp_shapes, args):
#     A = args[0]
#     dim = args[1]
#     idx = args[2]
#     import pdb; pdb.set_trace()
#     def f(*idxs):
#         # idxs = list(idxs)
#         idxs.insert(dim, to_expr(idx))
#         # idxs = [to_expr(0)]
#         return A.load(idxs)
#     res = te.Compute(name, get_dim_args(out_shape), f)
#     return res

def dot_lower(name, out_shape, inp_shapes, args):
    mul_te = te.lower('aten::mul', list(args), get_te_shapes(inp_shapes[0][0]), get_nnc_type(inp_shapes[0][1]))
    res = te.lower('aten::sum', [mul_te.buf()], get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    return (res.buf(), [mul_te.stmt(), res.stmt()])

def mv_lower(name, out_shape, inp_shapes, args):
    A = args[0]
    B = args[1]
    N, M = inp_shapes[0][0]

    def f(n, m):
        return A.load([n, m]) * B.load([m])
    # mm = te.Compute('mm', get_dim_args([N,M]), f)
    # out = te.Reduce(name, get_dim_args([N]), te.Sum(), mm, get_dim_args([M]))
    # return out.buf(), [mm.stmt(), out.stmt()]
    C = torch._C._te.BufHandle('C', get_te_shapes([N]), get_nnc_type(inp_shapes[0][1]))
    s = torch._C._te.ExternalCall(C, "nnc_aten_mv", [A, B], [])
    return C, [s]

lowering_functions[torch.ops.aten.ones_like] = ones_like_lower
lowering_functions[torch.ops.aten.dot] = dot_lower
# lowering_functions[torch.ops.aten.select] = select_lower
lowering_functions[torch.ops.aten.mv] = mv_lower

func_to_aten = {
    operator.getitem: torch.ops.aten.slice,
    operator.add: torch.ops.aten.add,
    operator.mul: torch.ops.aten.mul,
    torch.mul: torch.ops.aten.mul,
    torch.sin: torch.ops.aten.sin,
    torch.cos: torch.ops.aten.cos,
}


def process_shape(x):
    if len(x) == 0:
        return [1]
    return x

def lower_function(node, op, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: (process_shape(arg.meta['tensor_meta'].shape), arg.meta['tensor_meta'].dtype) if isinstance(arg, fx.Node) and 'tensor_meta' in arg.meta else None)
    if op in lowering_functions:
        out = lowering_functions[op](node.name, process_shape(node.meta['tensor_meta'].shape), inp_shapes, nnc_args)
    else:
        if op in func_to_aten:
            op = func_to_aten[op]
        aten_str = f'aten::{op.__name__}'
        out_shape = get_te_shapes(process_shape(node.meta['tensor_meta'].shape))
        out = te.lower(aten_str, list(nnc_args), out_shape, get_nnc_type(node.meta['tensor_meta'].dtype))
    if isinstance(out, te.Tensor):
        return out.buf(), [out.stmt()]
    else:
        return out[0], out[1]

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
        return get_nnc_type(node.meta['tensor_meta'].dtype)

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
            shapes = get_te_shapes(node.meta['tensor_meta'].shape)
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.data()
            inputs.append(placeholder)
        elif node.op == 'call_function':
            # This does the bulk of the work - we call `lower_function`, which
            # returns a `te.ExprHandle` (the output of a NNC computation), and
            # put it in our environment.
            if 'tensor_meta' in node.meta:
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
            outs = (list(te_args), [(i.meta['tensor_meta'].shape, i.meta['tensor_meta'].dtype) for i in args])
        elif node.op == 'get_attr':
            # As NNC doesn't have any concept of state, we pull out the module
            # attributes and pass them in as inputs to NNC.
            module_attrs.append(node)
            shapes = get_te_shapes(process_shape(node.meta['tensor_meta'].shape))
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.data()
        else:
            print(node.op, node.target)
            raise RuntimeError("not yet implemented")


    loopnest = te.LoopNest(te.Stmt(compute_stmts), outs[0])
    # loopnest.inline_intermediate_bufs(True)
    loopnest.simplify()
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + outs[0]])
    alloc_results = [torch.empty(shape, dtype=dtype) for shape,dtype in outs[1]]
    if module_attrs:
        module_stuff = [fetch_attr(i.target).contiguous().data for i in module_attrs]
    else:
        module_stuff = []
    def f(*inps, out_tensors=None):
        # begin = time.time()
        if out_tensors is None:
            results = alloc_results
        else:
            results = out_tensors
        cg.call(module_stuff + list(inps) + results)
        if out_tensors is None:
            if len(results) == 1:
                return results[0]
            return results
    return f


################################
# Example usage and Benchmarking
################################

def bench(f, warmup=3, iters=1000):
    for _ in range(warmup):
        f()
    begin = time.time()
    for _ in range(iters):
        f()
    print(time.time()-begin)
if __name__ == '__main__':
    def f(a, b):
        return (torch.cos(a)* torch.sin(b))[:2000]

    mod = fx.symbolic_trace(f)
    inps = (torch.randn(5000), torch.randn(5000))
    ShapeProp(mod).propagate(*inps)
    cg = nnc_compile(mod, inps)
    bench(lambda: cg(*inps))
    bench(lambda: f(*inps))
    exit(0)
import torch
torch._C._jit_override_can_fuse_on_cpu(True)
import torch.distributions as dist
from torch.autograd import grad
from nnc_compile import nnc_compile, decompose
import time

import jax
print('jax version: ', jax.__version__)

from collections import namedtuple
import numpy as onp
import jax
from vmap import grad as fx_grad
import torch.fx as fx
from jax import grad, jit, partial, random, value_and_grad, lax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax import random
from jax.tree_util import register_pytree_node, tree_multimap

# (q, p) -> (position (param value), momentum)

IntegratorState = namedtuple("IntegratorState", ["q", "p", "potential_energy", "q_grad"])

# a tree-like JAX primitive that allows program transformations
# to work on Python containers (https://jax.readthedocs.io/en/latest/pytrees.html)
register_pytree_node(
    IntegratorState,
    lambda xs: (tuple(xs), None),
    lambda _, xs: IntegratorState(*xs)
)


def leapfrog(potential_fn, kinetic_fn):
    r"""
    Second order symplectic integrator that uses the leapfrog algorithm
    for position `q` and momentum `p`.

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum.
    :return: a pair of (`init_fn`, `update_fn`).
    """
    def init_fn(q, p):
        """
        :param q: Position of the particle.
        :param p: Momentum of the particle.
        :return: initial state for the integrator.
        """
        potential_energy, q_grad = value_and_grad(potential_fn)(q)
        return IntegratorState(q, p, potential_energy, q_grad)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        :param float step_size: Size of a single step.
        :param inverse_mass_matrix: Inverse of mass matrix, which is used to
            calculate kinetic energy.
        :param state: Current state of the integrator.
        :return: new state for the integrator.
        """
        q, p, _, q_grad = state
        # maps a function over a pytree, returning a new pytree
        p = tree_multimap(lambda p, q_grad: p - 0.5 * step_size * q_grad, p, q_grad)  # p(n+1/2)
        p_grad = grad(kinetic_fn, argnums=1)(inverse_mass_matrix, p)
        q = tree_multimap(lambda q, p_grad: q + step_size * p_grad, q, p_grad)  # q(n+1)
        potential_energy, q_grad = value_and_grad(potential_fn)(q)
        p = tree_multimap(lambda p, q_grad: p - 0.5 * step_size * q_grad, p, q_grad)  # p(n+1)
        return IntegratorState(q, p, potential_energy, q_grad)

    return init_fn, update_fn


def kinetic_fn(inverse_mass_matrix, p):
    # flattens the pytree
    p, _ = ravel_pytree(p)

    if inverse_mass_matrix.ndim == 2:
        v = np.matmul(inverse_mass_matrix, p)
    elif inverse_mass_matrix.ndim == 1:
        v = np.multiply(inverse_mass_matrix, p)

    return 0.5 * np.dot(v, p)

D = 100

true_mean, true_std = np.ones(D), np.ones(D) * 2.

def potential_fn(q):
    """
    - log density for the normal distribution
    """
    return 0.5 * np.sum(((q['z'] - true_mean) / true_std) ** 2)


# U-turn termination condition
# For demonstration purpose - this won't result in a correct MCMC proposal
cnt = 0
def is_u_turning(q_i, q_f, p_f):
  return np.less(np.dot((q_f['z'] - q_i['z']), p_f['z']), 0)


# Run leapfrog until termination condition is met
def get_final_state(ke, pe, m_inv, step_size, q_i, p_i):
    lf_init, lf_update = leapfrog(pe, ke)
    x = lf_init(q_i, p_i)
    q_f, p_f, _, _ = lax.while_loop(lambda x: ~is_u_turning(q_i, x[0], x[1]),
                                    lambda x: lf_update(step_size, m_inv, x),
                                    x)
    return (q_f, p_f)


"""### jit and grad composition

Note that we are jit compiling the integrator which includes grad computation.
"""

# Commented out IPython magic to ensure Python compatibility.
q_i = {'z': np.zeros(D)}
p_i = lambda i: {'z': random.normal(random.PRNGKey(i), (D,))}
inv_mass_matrix = np.eye(D)
step_size = 0.001
seed = 1

fn = jit(get_final_state, static_argnums=(0, 1))
# fn = get_final_state
timefn = lambda i: fn(kinetic_fn, potential_fn, inv_mass_matrix,
                      step_size, q_i, p_i(i))

# Run only once in a loop; otherwise the best number reported
# does not include compilation time.
out = timefn(seed)[0]['z'].block_until_ready()

begin = time.time()

out = timefn(seed)
print(out[0]['z'][0])
print("jax time: ", time.time()-begin)


###########################
# PyTorch Start
###########################
MANUAL_GRAD = True
VERSION = 'nnc'
true_mean, true_std = 1, 2.

def potential_fn(params):
    return 0.5 * torch.sum(((params - true_mean) / true_std) ** 2.0)

if MANUAL_GRAD:
    def potential_grad(params):
        return potential_fn(params), (params-true_mean)/(true_std**2.0)
else:
    example_inps = (torch.randn(D),)
    potential_grad = fx.symbolic_trace(fx_grad(decompose(potential_fn, example_inps), example_inps))

"""## PyTorch NUTS example: LeapFrog Integrator"""
def leapfrog(q, p, potential_fn, inverse_mass_matrix, step_size):
    r"""
    Second order symplectic integrator that uses the velocity leapfrog algorithm.

    :param dict q: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict p: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given q
        for each sample site. The negative gradient of the function with respect
        to ``q`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param torch.Tensor inverse_mass_matrix: a tensor :math:`M^{-1}` which is used
        to calculate kinetic energy: :math:`E_{kinetic} = \frac{1}{2}z^T M^{-1} q`.
        Here :math:`M` can be a 1D tensor (diagonal matrix) or a 2D tensor (dense matrix).
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param torch.Tensor q_grads: optional gradients of potential energy at current ``q``.
    :return tuple (q_next, p_next, q_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``q_next``.
    """
    q_grads  = potential_grad(q)[1]

    p = p + 0.5*step_size*(-q_grads)

    p_grads = _kinetic_grad(inverse_mass_matrix, p)
    q = q + step_size * p_grads  # q(n+1)

    potential_energy, q_grads = potential_grad(q)
    p = p + 0.5*step_size*(-q_grads)

    return q, p, q_grads, potential_energy



def _kinetic_grad(inverse_mass_matrix, p):
    if inverse_mass_matrix.dim() == 1:
        grads = torch.mul(inverse_mass_matrix, p)
    else:
        grads = torch.matmul(inverse_mass_matrix, p)

    return grads





def is_u_turning(q_i, q_f, p_f):
  return torch.dot((q_f - q_i), p_f) < 0.

def get_final_state(pe, m_inv, step_size, q_i, p_i):
    q, p = q_i, p_i
    while not is_u_turning(q_i, q, p):
        # The function signature changes depending on whether we specialize the args.
        q, p, q_grads, _ = leapfrog(q, p)
        # q, p, q_grads, _ = leapfrog(q, p, pe, m_inv, step_size, q_grads=q_grads)

    return (q, p)



q_i =  torch.tensor(onp.array(q_i['z']))
p_i =  torch.tensor(onp.array(p_i(seed)['z']))
inv_mass_matrix = torch.eye(D)
# inv_mass_matrix = torch.ones(D)
step_size = 0.001
num_steps = 10000

import torch.fx as fx
is_u_turning = decompose(is_u_turning, example_inputs=(q_i, p_i, q_i))
is_u_turning = nnc_compile(is_u_turning, example_inputs=(q_i, p_i, q_i))

leapfrog = fx.symbolic_trace(leapfrog, concrete_args={'potential_fn': potential_fn, 'step_size': step_size, 'inverse_mass_matrix': inv_mass_matrix})
leapfrog = decompose(leapfrog, example_inputs=(q_i, p_i, q_i))
if VERSION == 'nnc':
    leapfrog = nnc_compile(leapfrog, example_inputs=(q_i, p_i, q_i))
elif VERSION == 'ts':
    leapfrog = torch.jit.script(leapfrog)
elif VERSION == 'pt':
    pass



out = get_final_state(potential_fn, inv_mass_matrix, step_size, q_i, p_i)
begin = time.time()
out = get_final_state(potential_fn, inv_mass_matrix, step_size, q_i, p_i)
print(out[0][0])
print("pytorch time: ", time.time()-begin)
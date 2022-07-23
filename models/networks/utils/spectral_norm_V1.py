"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import jittor
from jittor.misc import normalize
from typing import Any, Optional, TypeVar
from jittor.nn import Module
import numpy as np




class SpectralNorm:

    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12): # return None
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: jittor.Var): # return jittor.Var
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool): # return jittor.Var
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with jittor.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(jittor.nn.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = normalize(jittor.nn.matmul(weight_mat, v), dim=0, eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()
        setattr(module, self.name + '_u', u.stop_grad())
        setattr(module, self.name + '_v', v.stop_grad())
        sigma = jittor.sum(u * jittor.matmul(weight_mat, v)) 

        weight = weight / sigma
        assert jittor.misc.isnan(weight.flatten()[0]) == False
        return weight

    def remove(self, module: Module): 
        with jittor.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')

        setattr(module, self.name, jittor.Var(weight.detach()))

    def __call__(self, module: Module, inputs: Any): 
        a = self.compute_weight(module, do_power_iteration=module.is_training())
        #print(module, self.name, a.shape)
        setattr(module, self.name, a)

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = jittor.matmul(
                weight_mat.t().mm(weight_mat).pinverse(), 
                jittor.matmul(
                    weight_mat.t(), u.unsqueeze(1)
                    )
            ).squeeze(1)
        return v.mul_(target_sigma / jittor.matmul(u, jittor.matmul(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float): # return 'SpectralNorm'

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')

        with jittor.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            np.random.seed(42)
            _u = np.random.randn(h)
            u = normalize(jittor.Var(_u).stop_grad(), dim=0, eps=fn.eps)
            u.stop_grad()
            #v = normalize(jittor.randn([w]), dim=0, eps=fn.eps)
            np.random.seed(43)
            _v = np.random.randn(w)
            v = normalize(jittor.Var(_v).stop_grad(), dim=0, eps=fn.eps)
            v.stop_grad()

        delattr(module, fn.name)
        setattr(module, fn.name + "_orig", weight)
        weight_data = weight[:].stop_grad()
        setattr(module, fn.name, weight_data)
        setattr(module, fn.name + "_u", u)
        setattr(module, fn.name + "_v", v)

        module.register_pre_forward_hook(fn)
        return fn

T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None): # return T_module
    
    if dim is None:
        if isinstance(module, (jittor.nn.ConvTranspose,
                               jittor.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module



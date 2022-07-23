"""Implements the (unbiased) Sinkhorn divergence between sampled measures."""

import numpy as np
from functools import partial
from .utils import squared_distances, distances
from .sinkhorn_divergence import scaling_parameters
from .sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop


import jittor


cost_routines = {
    1 : (lambda x, y: distances(x, y)),
    2 : (lambda x, y: squared_distances(x, y) / 2)}


def logsumexp(x):
    #x = x.float64()
    c = x.max(2, True)
    c_squeeze = c.squeeze(2)
    return c_squeeze + (x - c).exp().sum(2,False).log()

def softmin_tensorized(ε, C, f):
    B = C.shape[0]
    
    a = (f.view(B, 1, -1) - C/ε)
    #print(a[0])
    a = logsumexp(a)
    #print(a[0])
    
    return - ε * a.view(B, -1)
    
    #return - ε * (f.view(B, 1, -1) - C/ε).logsumexp(2).view(B, -1)

def sinkhorn_tensorized(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, 
                        debias=True, potentials=False, **kwargs):
    
    B, N, D = x.shape
    _, M, _ = y.shape

    #print('sinkhorn_tensorized input ')

    #print('x',x.flatten()[:10])
    #print('y',y.flatten()[:10])

    if cost is None:
        cost = cost_routines[p]
        
    C_xx, C_yy = (cost(x, x.detach()), cost(y, y.detach())) if debias else (None, None)  # (B,N,N), (B,M,M)
    C_xy, C_yx = (cost(x, y.detach()), cost(y, x.detach()))  # (B,N,M), (B,M,N)

    #print('sinkhorn_loop input ')

    #print('C_xx',C_xx.flatten()[:10])
    #print('C_yy',C_yy.flatten()[:10])
    #print('C_xy',C_xy.flatten()[:10])
    #print('C_yx',C_yx.flatten()[:10])


    diameter, ε, ε_s, ρ = scaling_parameters(x, y, p, blur, reach, diameter, scaling)

    #print(diameter, ε, ε_s, ρ)

    a_x, b_y, a_y, b_x = sinkhorn_loop(softmin_tensorized, log_weights(α), log_weights(β),
                                       C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=debias)

    
    #print('sinkhorn_loop')

    #print(a_y.flatten()[:10])
    #print(b_x.flatten()[:10])

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=debias, potentials=potentials)
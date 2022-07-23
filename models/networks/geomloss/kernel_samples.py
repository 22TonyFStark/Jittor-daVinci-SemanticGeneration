"""Implements kernel ("gaussian", "laplacian", "energy") norms between sampled measures.
.. math::
    \\text{Loss}(\\alpha,\\beta) 
        ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\alpha_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\beta_j \,\delta_{y_j} \\big) 
        ~=~ \\tfrac{1}{2} \|\\alpha-\\beta\|_k^2 \\\\
        &=~ \\tfrac{1}{2} \langle \\alpha-\\beta \,,\, k\star (\\alpha - \\beta) \\rangle \\\\
        &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\alpha_i \\alpha_j \cdot k(x_i,x_j) 
          + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\beta_i \\beta_j \cdot k(y_i,y_j) \\\\
        &-~\sum_{i=1}^N \sum_{j=1}^M  \\alpha_i \\beta_j \cdot k(x_i,y_j)
where:
.. math::
    k(x,y)~=~\\begin{cases}
        \exp( -\|x-y\|^2/2\sigma^2) & \\text{if loss = ``gaussian''} \\\\
        \exp( -\|x-y\|/\sigma) & \\text{if loss = ``laplacian''} \\\\
        -\|x-y\| & \\text{if loss = ``energy''} \\\\
    \\end{cases}
"""

import numpy as np
import jittor
try:  
    keops_available = True
except:
    keops_available = False

from utils import scal, squared_distances, distances

class DoubleGrad(jittor.Function):
    @staticmethod
    def execute(ctx, input):
        return input

    @staticmethod
    def grad(ctx, grad_output):
        return 2*grad_output

def double_grad(x):
    return DoubleGrad.apply(x)


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def gaussian_kernel(x, y, blur=.05):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def laplacian_kernel(x, y, blur=.05):
    C = distances(x / blur, y / blur)
    return (- C ).exp()

def energy_kernel(x, y, blur=None):
    return - distances(x, y)

kernel_routines = {
    "gaussian" : gaussian_kernel,
    "laplacian": laplacian_kernel,
    "energy"   : energy_kernel,
}



# ==============================================================================
#                           backend == "online"
# ==============================================================================

kernel_formulas = {
    "gaussian" : ("Exp(-SqDist(X,Y) / IntCst(2))", True ),
    "laplacian": ("Exp(-Norm2(X-Y))",   True ),
    "energy"   : ("(-Norm2(X-Y))",      False),
}


def kernel_keops(kernel, α, x, β, y, potentials=False, ranges_xx = None, ranges_yy = None, ranges_xy = None):

    D = x.shape[1]
    kernel_conv = generic_sum( "(" + kernel + " * B)",   # Formula
                               "A = Vi(1)",              # Output:    a_i
                               "X = Vi({})".format(D),   # 1st input: x_i
                               "Y = Vj({})".format(D),   # 2nd input: y_j
                               "B = Vj(1)" )             # 3rd input: b_j
    
    a_x = kernel_conv(double_grad(x), x.detach(), α.detach().view(-1,1), ranges=ranges_xx)
    b_y = kernel_conv(double_grad(y), y.detach(), β.detach().view(-1,1), ranges=ranges_yy)
    b_x = kernel_conv(x, y, β.view(-1,1), ranges=ranges_xy)

    if potentials:
        a_y = kernel_conv(y, x, α.view(-1,1), ranges=swap_axes(ranges_xy))
        return a_x - b_x, b_y - a_y

    else:  # Return the Kernel norm. N.B.: we assume that 'kernel' is symmetric:
        return .5 * scal( double_grad(α), a_x ) \
             + .5 * scal( double_grad(β), b_y )  -  scal( α, b_x )
              


def kernel_preprocess(kernel, name, x, y, blur):
    if not keops_available:
        raise ImportError("The 'pykeops' library could not be loaded: " \
                        + "'online' and 'multiscale' backends are not available.")
    
    if kernel is None: kernel, rescale = kernel_formulas[name]
    else:              rescale = True
    
    # Center the point clouds just in case, to prevent numeric overflows:
    center = (x.mean(0, keepdim=True) + y.mean(0,  keepdim=True)) / 2
    x, y = x - center, y - center
    # Rescaling on x and y is cheaper than doing it for all pairs of points 
    if rescale : x, y = x / blur, y / blur
    
    return kernel, x, y


def kernel_online(α, x, β, y, blur=.05, kernel=None, name=None, potentials=False, **kwargs):

    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)
    return kernel_keops(kernel, α, x, β, y, potentials=potentials)


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================



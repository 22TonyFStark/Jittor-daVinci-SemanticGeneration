import jittor.nn as nn
import jittor



def hann_filter(kernel_size):
    r"""Creates  Hann kernel
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    # Take bigger window and drop borders
    window = jittor.Var([0.5000, 1.0000, 0.5000]).stop_grad()
    kernel = window[:, None] * window[None, :]
    # Normalize and reshape kernel
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()



class L2Pool2d(nn.Module):
    r"""Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = None

    def execute(self, x):
        if self.kernel is None:
            C = x.size(1)
            self.kernel = hann_filter(self.kernel_size).repeat((C, 1, 1, 1))
            

        out = nn.conv2d(
            x ** 2, self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1]
        )
        return (out + self.EPS).sqrt()


def similarity_map(map_x, map_y, constant, alpha=0.0):
    r""" Compute similarity_map between two tensors using Dice-like equation.
    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
        alpha: Masking coefficient. Subtracts - `alpha` * map_x * map_y from denominator and nominator
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / \
           (map_x ** 2 + map_y ** 2 - alpha * map_x * map_y + constant)
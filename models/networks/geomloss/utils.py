import jittor
import jittor.nn as nn
from jittor import Function


def scal(α, f, batch=False):
    if batch:
        B = α.shape[0]
        return (α.view(B, -1) * f.view(B, -1)).sum(1)
    else:
        return (α.view(-1) * f.view(-1)).sum()


class Sqrt0(Function):

    def execute(self, input):
        result = input.sqrt()
        result[input < 0] = 0
        self.saved_tensors = result
        return result


    def grad(self, grad_output):
        #result, = self.saved_tensors
        result = self.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    func = Sqrt0.apply
    return func(x)


def squared_distances(x, y):
    D = 1 - nn.matmul(x, y.permute(0, 2, 1))
    return D

def distances(x, y):
    return sqrt_0(squared_distances(x, y))

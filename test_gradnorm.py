import jittor
import numpy as np

np.random.seed(1)
x = jittor.Var(np.random.randn(2,3,64,64))

f = jittor.nn.Conv2d(3,5,1,bias=False)
np.random.seed(2)
f.weight = jittor.Var(np.random.randn(5,3,1,1))

def normalize_gradient(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    #x.requires_grad_(True)
    assert x.requires_grad == True
    f = net_D(x, **kwargs)
    grad = jittor.grad(
        f, [x])[0]
    grad_norm = jittor.norm(jittor.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + jittor.abs(f)))
    return f_hat

out = normalize_gradient(f, x)


print(out.flatten()[::1024])
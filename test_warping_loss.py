import numpy as np
import jittor
from models.networks.geomloss.samples_loss import SamplesLoss
if __name__ == '__main__':
    p, blur = 1, 0.025#0.005
    uot = SamplesLoss("sinkhorn", p=p, blur=blur,
                                      debias=False, potentials=True)
    np.random.seed(1)
    inp_a = np.random.randn(6,4096)
    np.random.seed(2)
    inp_b = np.random.randn(6,4096, 2304)
    np.random.seed(3)
    inp_c = np.random.randn(6,4096)
    np.random.seed(4)
    inp_d = np.random.randn(6,4096, 2304)
    a = jittor.Var(inp_a)

    b = jittor.Var(inp_b)

    c = jittor.Var(inp_c)

    d = jittor.Var(inp_d)
    F,G = uot(a,b,c,d)
    print(inp_a.flatten()[:3],inp_b.flatten()[:3],inp_c.flatten()[:3],inp_d.flatten()[:3])
    print(F.flatten()[::100])
    print(G.flatten()[::100])
    print('finish')
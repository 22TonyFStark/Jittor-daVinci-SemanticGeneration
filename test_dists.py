import jittor.models as models
import jittor
net = models.vgg16(pretrained=False).features
for param in net.parameters():
    param.stop_grad()
print()
for name, module in net.named_parameters():
    print(name, module.shape,module.requires_grad)
net = net.requires_grad_(False)
for name, module in net.named_parameters():
    print(name, module.shape,module.requires_grad)
"""
def replace_pooling(module):

    new_net = jittor.nn.Sequential()
    
    for name, child in module.named_modules()[1:]:
        print(child)
        if isinstance(child, jittor.nn.Pool):
            child = jittor.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        new_net.add_module(name, child)
    return new_net
    
net = replace_pooling(net)

print(net)
"""
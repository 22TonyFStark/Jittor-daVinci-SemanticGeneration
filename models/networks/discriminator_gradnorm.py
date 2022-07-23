"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor
import jittor.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import Attention
import util.util as util

def normalize_gradient_lastonly(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    #x.requires_grad_(True)
    assert x.requires_grad == True
    fs, cam_logit = net_D(x, **kwargs)
    grad = jittor.grad(
        fs[-1], [x])[0]
    grad_norm = jittor.norm(jittor.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(fs[-1].shape) - 1)])
    fs[-1] = (fs[-1] / (grad_norm + jittor.abs(fs[-1])))
    return fs, None

def normalize_gradient_multi(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    #x.requires_grad_(True)
    assert x.requires_grad == True
    fs, cam_logit = net_D(x, **kwargs)
    f_hats = []
    for i in range(len(fs)):
        f = fs[i]
        grad = jittor.grad(
            f, [x])[0]
        grad_norm = jittor.norm(jittor.flatten(grad, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
        f_hat = (f / (grad_norm + jittor.abs(f)))
        f_hats.append(f_hat)
    return f_hats, None

class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()


        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')

        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, stage1=False):
        super().__init__()
        self.opt = opt
        self.stage1 = stage1

        self.num_D = opt.num_D

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            setattr(self, 'discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, stage1=self.stage1)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return nn.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    def execute(self, input):
        result = []
        segs = []
        cam_logits = []
        get_intermediate_features = not self.opt.no_ganFeat_loss

        for i in range(self.num_D):
            #print(i)
            #print('*'*50)
            D = getattr(self, 'discriminator_%d' % i)
            """ 梯度归一化 """  
            #out, cam_logit = D(input)
            
            #out, cam_logit = normalize_gradient_multi(D, input)
            
            out, cam_logit = normalize_gradient_lastonly(D, input)

            cam_logits.append(cam_logit)
            if not get_intermediate_features:
                out = [out]
            result.append(out)

            try:
                input = self.downsample(input)
            except:
                print(input.shape)
                for inp in input:
                    print(inp.shape)
                raise "s"
            #print(i,'out', out[0].flatten()[::10000])
        


        return result, segs, cam_logits


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, stage1=False):
        super().__init__()
        self.opt = opt
        self.stage1 = stage1

        kw = 4
        #padw = int(np.ceil((kw - 1.0) / 2))
        padw = int((kw - 1.0) / 2)
        nf = opt.ndf
        # input_nc = self.compute_D_input_nc(opt)
        input_nc = opt.semantic_nc + 3

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     jittor.nn.LeakyReLU(0.2)
                     ]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            if (((not stage1) and opt.use_attention) or (stage1 and opt.use_attention_st1)) and n == opt.n_layers_D - 1:
                self.attn = Attention(nf_prev, 'spectral' in opt.norm_D)
            if n == opt.n_layers_D - 1 and (not stage1):
                dec = []
                nc_dec = nf_prev
                for _ in range(opt.n_layers_D - 1):
                    dec += [nn.Upsample(scale_factor=2),
                            norm_layer(nn.Conv2d(nc_dec, int(nc_dec//2), kernel_size=3, stride=1, padding=1)),
                            jittor.nn.LeakyReLU(0.2)
                            ]
                    nc_dec = int(nc_dec // 2)
                dec += [nn.Conv2d(nc_dec, opt.semantic_nc, kernel_size=3, stride=1, padding=1)]
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          jittor.nn.LeakyReLU(0.2) 
                          ]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]


        if opt.D_cam > 0:
            mult = min(2 ** (opt.n_layers_D - 1), 8)

            self.conv1x1 = nn.Conv2d(opt.ndf * mult * 2, opt.ndf * mult, kernel_size=1, stride=1, bias=True)
            self.leaky_relu = nn.LeakyReLU(0.2, True)

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

        self.n_model = len(sequence)

        #self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        return input_nc

    def execute(self, input):
        #print(input.shape) [2,32,256,256,]
        results = [input]
        seg = None
        cam_logit = None

        for name, submodel in self.named_modules():
            if '.' in name:
                continue
            if 'model' not in name:
                continue

            if name == 'model3':
                if ((not self.stage1) and self.opt.use_attention) or (self.stage1 and self.opt.use_attention_st1):
                    x = self.attn(results[-1])
                else:
                    x = results[-1]
            else:
                x = results[-1]
            intermediate_output = submodel(x)
            if self.opt.D_cam > 0 and name == 'model3.0':  
                gap = nn.adaptive_avg_pool2d(intermediate_output, 1)
                gap_logit = self.gap_fc(gap.view(intermediate_output.shape[0], -1))
                gap_weight = list(self.gap_fc.parameters())[0]
                gap = intermediate_output * gap_weight.unsqueeze(2).unsqueeze(3)

                gmp = nn.adaptive_max_pool2d(intermediate_output, 1)
                gmp_logit = self.gmp_fc(gmp.view(intermediate_output.shape[0], -1))
                gmp_weight = list(self.gmp_fc.parameters())[0]
                gmp = intermediate_output * gmp_weight.unsqueeze(2).unsqueeze(3)

                cam_logit = jittor.concat([gap_logit, gmp_logit], 1)
                intermediate_output = jittor.concat([gap, gmp], 1)
                intermediate_output = self.leaky_relu(self.conv1x1(intermediate_output))
            
            results.append(intermediate_output)

            # print('********')
            # print(x.shape)
        # print(1 / 0)




        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]
        if seg is None:
            return retu, cam_logit
        else:
            return retu, seg, cam_logit

def find_network_using_name(target_network_name, filename, add=True):
    target_class_name = target_network_name + filename if add else target_network_name
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
       "Class %s should be a subclass of BaseNetwork" % network

    return network

def test():

    ########### test opt ############
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')

    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
    parser.add_argument('--use_attention', action='store_true', help='and nonlocal block in G and D')
    parser.add_argument('--eqlr_sn', action='store_true', help='if true, use equlr, else use sn')
    parser.add_argument('--D_cam', type=float, default=0.0, help='weight of CAM loss in D')
    opt = parser.parse_args()

    opt.semantic_nc = 29
    opt.num_D = 2
    opt.netD_subarch = 'n_layer'
    

    D_test = MultiscaleDiscriminator(opt)
    print(D_test)

if __name__ == '__main__':
    test()
    pass

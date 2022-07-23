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

""" Sesame """

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
        
        print("using sesame D")

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
            out, cam_logit = D(input)
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
        
        # 分为两个branch，一个是语义图=input_nc，一个是RGB
        branch = []
        sizes = (input_nc - 3, 3) 
        original_nf = nf
        
        for input_nc in sizes:
            nf = original_nf
            
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         jittor.nn.LeakyReLU(0.2)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                if (((not stage1) and opt.use_attention) or (stage1 and opt.use_attention_st1)) and n == opt.n_layers_D - 1:
                    self.attn = Attention(nf_prev, 'spectral' in opt.norm_D)

                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                stride=stride, padding=padw)),
                            jittor.nn.LeakyReLU(0.2)
                            ]]

            branch.append(sequence)

        # 语义图的分支
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)
        
        
        # RGB的分支
        sequence = branch[1]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]


        if opt.D_cam > 0:
            mult = min(2 ** (opt.n_layers_D - 1), 8)
            if opt.eqlr_sn:
                self.gap_fc = equal_lr(nn.Linear(opt.ndf * mult, 1, bias=False))
                self.gmp_fc = equal_lr(nn.Linear(opt.ndf * mult, 1, bias=False))
            else:
                self.gap_fc = nn.utils.spectral_norm(nn.Linear(opt.ndf * mult, 1, bias=False))
                self.gmp_fc = nn.utils.spectral_norm(nn.Linear(opt.ndf * mult, 1, bias=False))
            self.conv1x1 = nn.Conv2d(opt.ndf * mult * 2, opt.ndf * mult, kernel_size=1, stride=1, bias=True)
            self.leaky_relu = nn.LeakyReLU(0.2, True)

        # 这是RGB分支，model命名
        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence_num = len(sequence)
        for n in range(self.img_sequence_num):
            #self.add_module('model' + str(n), nn.Sequential(*sequence[n]))
            #self.layers['model' + str(n)] = nn.Sequential(*sequence[n])
            setattr(self, 'img_sequence' + str(n), nn.Sequential(*sequence[n]))
            

        self.n_model = len(sequence)

        #self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        return input_nc

    def execute(self, input):
        #print(input.shape) [2,32,256,256,]
        img, sem = input[:,-3:], input[:,:-3]
        #print("sem input shape: ", sem.shape) # [4,29,256,256,]
        sem_results = self.sem_sequence(sem) # [4,512,31,31,]
        #print("sem output shape: ", sem_results.shape)
        results = [img]
        
        # results = [input]
        seg = None
        cam_logit = None
        
        """
        discriminator_1: NLayerDiscriminator(
                sem_sequence: Sequential(
                    0: Sequential(
                        0: Conv(29, 64, (4, 4), (2, 2), (1, 1), (1, 1), 1, float32[64,], None, Kw=None, fan=None, i=None, bound=None)
                        1: leaky_relu(0.2)
                    )
                    1: Sequential(
                        0: Sequential(
                            0: Conv(64, 128, (4, 4), (2, 2), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                            1: InstanceNorm(128, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                        )
                        1: leaky_relu(0.2)
                    )
                    2: Sequential(
                        0: Sequential(
                            0: Conv(128, 256, (4, 4), (2, 2), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                            1: InstanceNorm(256, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                        )
                        1: leaky_relu(0.2)
                    )
                    3: Sequential(
                        0: Sequential(
                            0: Conv(256, 512, (4, 4), (1, 1), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                            1: InstanceNorm(512, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                        )
                        1: leaky_relu(0.2)
                    )
                )
                img_sequence0: Sequential(
                    0: Conv(3, 64, (4, 4), (2, 2), (1, 1), (1, 1), 1, float32[64,], None, Kw=None, fan=None, i=None, bound=None)
                    1: leaky_relu(0.2)
                )
                img_sequence1: Sequential(
                    0: Sequential(
                        0: Conv(64, 128, (4, 4), (2, 2), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                        1: InstanceNorm(128, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                    )
                    1: leaky_relu(0.2)
                )
                img_sequence2: Sequential(
                    0: Sequential(
                        0: Conv(128, 256, (4, 4), (2, 2), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                        1: InstanceNorm(256, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                    )
                    1: leaky_relu(0.2)
                )
                img_sequence3: Sequential(
                    0: Sequential(
                        0: Conv(256, 512, (4, 4), (1, 1), (1, 1), (1, 1), 1, None, None, Kw=None, fan=None, i=None, bound=None)
                        1: InstanceNorm(512, 1e-05, momentum=0.1, affine=False, is_train=True, sync=True)
                    )
                    1: leaky_relu(0.2)
                )
                img_sequence4: Sequential(
                    0: Conv(512, 1, (4, 4), (1, 1), (1, 1), (1, 1), 1, float32[1,], None, Kw=None, fan=None, i=None, bound=None)
                )
            )
        """

        for layer_id in range(self.opt.n_layers_D):
            submodel = getattr(self, 'img_sequence'+str(layer_id))

            if layer_id == 3:
                if ((not self.stage1) and self.opt.use_attention) or (self.stage1 and self.opt.use_attention_st1):
                    x = self.attn(results[-1])
                else:
                    x = results[-1]
            else:
                x = results[-1]
            intermediate_output = submodel(x)
            if self.opt.D_cam > 0 and layer_id == 3:
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
    
    
        # 语义图分支
        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence4(intermediate_output))

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]
        if seg is None:
            return retu, cam_logit
        else:
            return retu, seg, cam_logit
        
    
    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)



def find_network_using_name(target_network_name, filename, add=True):
    target_class_name = target_network_name + filename if add else target_network_name
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
       "Class %s should be a subclass of BaseNetwork" % network

    return network

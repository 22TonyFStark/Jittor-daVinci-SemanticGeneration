# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#08.09 change pad

import numpy as np
import jittor.nn as nn
import jittor
from jittor import Function

from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SEACEResnetBlock as SEACEResnetBlock
from models.networks.architecture import Ada_SPADEResnetBlock as Ada_SPADEResnetBlock
from models.networks.architecture import Attention


class SEACEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = 64
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        ic = opt.semantic_nc
        self.fc = nn.Conv2d(16 * nf, 16 * nf, 3, stride=1, padding=1)
        self.G_head_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=False)

        self.G_middle_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=True)
        self.G_middle_1 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=False)

        self.G_up_0 = SEACEResnetBlock(16 * nf, 8 * nf, opt, feat_nc=256, atten=True)
        self.G_up_1 = SEACEResnetBlock(8 * nf, 4 * nf, opt, feat_nc=256, atten=False)
        self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)

        self.G_out_0 = SEACEResnetBlock(4 * nf, 2 * nf, opt, feat_nc=128, atten=True)
        self.G_out_1 = SEACEResnetBlock(2 * nf, 1 * nf, opt, feat_nc=3, atten=False)

        self.conv_img1 = nn.Conv2d(1 * nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.attn2 = Self_Attn(128, 'relu')
        self.attn3 = Self_Attn(256, 'relu')
        self.attn4 = Self_Attn(512, 'relu')

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def execute(self, warp_out=None):

        seg_feat1, seg_feat2, seg_feat3, seg_feat4, seg_feat5, \
        ref_feat1, ref_feat2, ref_feat3, ref_feat4, ref_feat5, conf_map = warp_out
        #  3, 128, 256, 512, 512


        #atten2 = self.attn2(seg_feat2, size=64)
        #atten3 = self.attn3(seg_feat3, size=32)
        #atten4 = self.attn4(seg_feat4, size=16)
        atten2 = self.attn2(seg_feat2, size=self.opt.crop_size // 4)
        atten3 = self.attn3(seg_feat3, size=self.opt.crop_size // 8)
        atten4 = self.attn4(seg_feat4, size=self.opt.crop_size // 16)

        x = jittor.concat((seg_feat5, ref_feat5), 1)
        x = nn.interpolate(x, size=(self.sh, self.sw),mode='nearest')

        x = self.fc(x)

        x = self.G_head_0(x, seg_feat5, ref_feat5, None, conf_map)
        x = self.up(x)

        x = self.G_middle_0(x, seg_feat4, ref_feat4, atten4, conf_map) # 16
        x = self.G_middle_1(x, seg_feat4, ref_feat4, None, conf_map)
        x = self.up(x)

        x = self.G_up_0(x, seg_feat3, ref_feat3, atten3, conf_map) # 32
        x = self.up(x)
        x = self.G_up_1(x, seg_feat3, ref_feat3, None, conf_map)
        x = self.up(x)

        x = self.attn(x) # 128,
        x = self.G_out_0(x, seg_feat2, ref_feat2, atten2, conf_map)
        x = self.up(x)
        x = self.G_out_1(x, seg_feat1, ref_feat1, None, conf_map)

        x = self.conv_img1(nn.leaky_relu(x, 2e-1))
        x = jittor.tanh(x)
        #print('G out')
        #print(x.flatten()[0:1000:100])


        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x, size):
        x = nn.interpolate(x, size=(size, size), mode='nearest')
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)

        energy = nn.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        return attention




class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        nf = 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.opt = opt
        self.head_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)

        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)

        self.deeper2 = Ada_SPADEResnetBlock(8 * nf, 4 * nf, opt, dilation=4)
        self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))


    def execute(self, input, seg, multi=False,debug=False):
        
        x = self.layer1(input)

        x = self.layer2(nn.leaky_relu(x, 0.2))
        x2 = x

        x = self.layer3(nn.leaky_relu(x, 0.2))
        x3 = x
        if debug:
            print(x.flatten()[::1024])


        x = self.layer4(nn.leaky_relu(x, 0.2))

        if debug:
            print(x.flatten()[::1024])

        x = self.layer5(nn.leaky_relu(x, 0.2))

        if debug:
            print(x.flatten()[::1024])

        x = self.head_0(x, seg)

        if debug:
            print(x.flatten()[::1024])

            raise "check adaptive "
        x4 = x



        # x = self.head_1(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x5 = x

        # x = self.deeper0(x, seg)
        # x = self.deeper1(x, seg)
        x = self.deeper2(x, seg)
        x = self.degridding0(x)
        # x = self.degridding1(x)

        if multi == True:
            return x2, x3, x4, x5, x
        else:
            return x


class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
                                nn.BatchNorm2d(2 * nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
                                nn.BatchNorm2d(nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
                                nn.BatchNorm2d(int(nf // 2), affine=True),
                                nn.LeakyReLU(0.2, False))  #32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100),
                nn.BatchNorm2d(100, affine=True),
                nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2),
                      nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def execute(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]



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

    parser.add_argument('--netG', type=str, default='seace', help='selects model to use for netG (pix2pixhd | spade | seace)')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--z_dim', type=int, default=256,
                        help="dimension of the latent z vector")

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.set_defaults(norm_G='spectralspadebatch3x3')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # for instance-wise features
    parser.add_argument('--CBN_intype', type=str, default='warp_mask', help='type of CBN input for framework, warp/mask/warp_mask')
    parser.add_argument('--maskmix', action='store_true', help='use mask in correspondence net')
    parser.add_argument('--use_attention', action='store_true', help='and nonlocal block in G and D')
    parser.add_argument('--warp_mask_losstype', type=str, default='none', help='type of warped mask loss, none/direct/cycle')
    parser.add_argument('--show_warpmask', action='store_true', help='save warp mask')
    parser.add_argument('--match_kernel', type=int, default=3, help='correspondence matrix match kernel size')
    parser.add_argument('--adaptor_kernel', type=int, default=3, help='kernel size in domain adaptor')
    parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
    parser.add_argument('--PONO_C', action='store_true', help='use C normalization in corr module')
    parser.add_argument('--eqlr_sn', action='store_true', help='if true, use equlr, else use sn')
    parser.add_argument('--vgg_normal_correct', action='store_true', help='if true, correct vgg normalization and replace vgg FM model with ctx model')
    parser.add_argument('--weight_domainC', type=float, default=0.0, help='weight of Domain classification loss for domain adaptation')
    parser.add_argument('--domain_rela', action='store_true', help='if true, use Relativistic loss in domain classifier')
    parser.add_argument('--use_ema', action='store_true', help='if true, use EMA in G')
    parser.add_argument('--ema_beta', type=float, default=0.999, help='beta in ema setting') 
    parser.add_argument('--warp_cycle_w', type=float, default=0.0, help='push warp cycle to ref')
    parser.add_argument('--two_cycle', action='store_true', help='input to ref and back')
    parser.add_argument('--apex', action='store_true', help='if true, use apex')
    parser.add_argument('--warp_bilinear', action='store_true', help='if true, upsample warp by bilinear')
    parser.add_argument('--adaptor_res_deeper', action='store_true', help='if true, use 6 res block in domain adaptor')
    parser.add_argument('--adaptor_nonlocal', action='store_true', help='if true, use nonlocal block in domain adaptor')
    parser.add_argument('--adaptor_se', action='store_true', help='if true, use se layer in domain adaptor')
    parser.add_argument('--dilation_conv', action='store_true', help='if true, use dilation conv in domain adaptor when adaptor_res_deeper is True')
    parser.add_argument('--use_coordconv', action='store_true', help='if true, use coordconv in CorrNet')
    parser.add_argument('--warp_patch', action='store_true', help='use corr matrix to warp 4*4 patch')
    parser.add_argument('--warp_stride', type=int, default=4, help='corr matrix 256 / warp_stride')
    parser.add_argument('--mask_noise', action='store_true', help='use noise with mask')
    parser.add_argument('--noise_for_mask', action='store_true', help='replace mask with noise')
    parser.add_argument('--video_like', action='store_true', help='useful in deepfashion')

    opt = parser.parse_args()

    opt.semantic_nc = 29
    opt.num_D = 2
    opt.netD_subarch = 'n_layer'
    

    G_test = SEACEGenerator(opt)
    print(G_test)

if __name__ == '__main__':
    test()
    pass

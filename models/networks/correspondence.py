import sys
import jittor
import jittor.nn as nn
from jittor import models
from jittor import Function
from models.networks.base_network import BaseNetwork
from models.networks.generator import AdaptiveFeatureGenerator, DomainClassifier
from util.util import vgg_preprocess
import util.util as util
from .geomloss import SamplesLoss
from .nceloss import BidirectionalNCE1

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def execute(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out

class WTA_scale(Function):


    @staticmethod
    def execute(ctx, input, scale=1e-4):
        """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
        activation_max, index_max = jittor.max(input, -1, keepdims=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = jittor.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(jittor.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def grad(ctx, grad_output):
        """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = jittor.ones_like(mask)
        mask_ones.stop_grad()
        mask_small_ones = jittor.ones_like(mask) * 1e-4
        mask_small_ones.stop_grad()

        grad_scale = jittor.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None




class VGG19_feature_color(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def execute(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = nn.relu(self.conv1_1(x))
        out['r12'] = nn.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = nn.relu(self.conv2_1(out['p1']))
        out['r22'] = nn.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = nn.relu(self.conv3_1(out['p2']))
        out['r32'] = nn.relu(self.conv3_2(out['r31']))
        out['r33'] = nn.relu(self.conv3_3(out['r32']))
        out['r34'] = nn.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = nn.relu(self.conv4_1(out['p3']))
        out['r42'] = nn.relu(self.conv4_2(out['r41']))
        out['r43'] = nn.relu(self.conv4_3(out['r42']))
        out['r44'] = nn.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = nn.relu(self.conv5_1(out['p4']))
        out['r52'] = nn.relu(self.conv5_2(out['r51']))
        out['r53'] = nn.relu(self.conv5_3(out['r52']))
        out['r54'] = nn.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]




class VGG19_feature_color_jittor(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_jittor, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        _vgg = models.vgg19(pretrained=True).features

        self.conv1_1 = _vgg[0]
        self.conv1_2 = _vgg[2]
        self.conv2_1 = _vgg[5]
        self.conv2_2 = _vgg[7]
        self.conv3_1 = _vgg[10]
        self.conv3_2 = _vgg[12]
        self.conv3_3 = _vgg[14]
        self.conv3_4 = _vgg[16]
        self.conv4_1 = _vgg[19]
        self.conv4_2 = _vgg[21]
        self.conv4_3 = _vgg[23]
        self.conv4_4 = _vgg[25]
        self.conv5_1 = _vgg[28]
        self.conv5_2 = _vgg[30]
        self.conv5_3 = _vgg[32]
        self.conv5_4 = _vgg[34]
        if pool == 'max':
            self.pool1 = _vgg[4]
            self.pool2 = _vgg[9]
            self.pool3 = _vgg[18]
            self.pool4 = _vgg[27]
            self.pool5 = _vgg[36]
        elif pool == 'avg':
            raise "NotImplementedError, by default using max-pooling"

        del _vgg

    def execute(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = nn.relu(self.conv1_1(x))
        out['r12'] = nn.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = nn.relu(self.conv2_1(out['p1']))
        out['r22'] = nn.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = nn.relu(self.conv3_1(out['p2']))
        out['r32'] = nn.relu(self.conv3_2(out['r31']))
        out['r33'] = nn.relu(self.conv3_3(out['r32']))
        out['r34'] = nn.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = nn.relu(self.conv4_1(out['p3']))
        out['r42'] = nn.relu(self.conv4_2(out['r41']))
        out['r43'] = nn.relu(self.conv4_3(out['r42']))
        out['r44'] = nn.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = nn.relu(self.conv5_1(out['p4']))
        out['r52'] = nn.relu(self.conv5_2(out['r51']))
        out['r53'] = nn.relu(self.conv5_3(out['r52']))
        out['r54'] = nn.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def execute(self, x):
        norm = x.pow(self.power).sum(1, keepdims=True).pow(1. / self.power)
        out = x.divide(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)

    def execute(self, feat, num_patches=64, patch_ids=None):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if patch_ids is not None:
            patch_id = patch_ids
        else:
            patch_id = jittor.randperm(feat_reshape.shape[1])
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        x_sample = self.l2norm(x_sample)
        # return_feats.append(x_sample)
        return x_sample, patch_id
    

class NoVGGCorrespondence(BaseNetwork):
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        # self.p, self.blur = 1, opt.blur
        self.p, self.blur = 1, 0.025#0.005
        self.uot = SamplesLoss("sinkhorn", p=self.p, blur=self.blur,
                                      debias=False, potentials=True)

        opt.spade_ic = opt.semantic_nc
        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt)
        opt.spade_ic = 3
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt)
        del opt.spade_ic
        if opt.weight_domainC > 0 and (not opt.domain_rela):
            self.domain_classifier = DomainClassifier(opt)


        self.down = opt.warp_stride # 4

        self.feat_ch = 64
        self.cor_dim = 256
        label_nc = opt.semantic_nc if opt.maskmix else 0
        coord_c = 3 if opt.use_coordconv else 0

        self.layer = nn.Sequential(
            ResidualBlock(self.feat_ch * 4 + label_nc + coord_c, self.feat_ch * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + label_nc + coord_c, self.feat_ch * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + label_nc + coord_c, self.feat_ch * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + label_nc + coord_c, self.feat_ch * 4 + label_nc + coord_c, kernel_size=3, padding=1, stride=1))

        self.phi = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)

        self.phi_w = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta_w = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)

        self.phi_conf = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta_conf = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        # self.theta_atten = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # self.upsampling_bi = F.interpolate(scale_factor=self.down, mode='bilinear')
        if opt.warp_bilinear:
            self.upsampling = nn.Upsample(scale_factor=self.down, mode='bilinear')
        else:
            self.upsampling = nn.Upsample(scale_factor=self.down)
        self.relu = nn.ReLU()
        
        self.nceloss = BidirectionalNCE1()
        self.patch_sample = PatchSampleF()
        

    def execute(self, ref_img, real_img, seg_map, ref_seg_map, detach_flag=False):
        coor_out = {}
        batch_size, _, im_height, im_width = ref_img.shape
        feat_height, feat_width = int(im_height / self.down), int(im_width / self.down)

        # 语义图提取的特征
        seg_feat2, seg_feat3, seg_feat4, seg_feat5, seg_feat6 = self.adaptive_model_seg(seg_map, seg_map, multi=True, debug=False)
        # 参考图提取的特征

        ref_feat2, ref_feat3, ref_feat4, ref_feat5, ref_feat6 = self.adaptive_model_img(ref_img, ref_img, multi=True)



        # 语义图的特征
        adp_feat_seg = util.feature_normalize(seg_feat6)
        # 参考图的特征
        

        adp_feat_img = util.feature_normalize(ref_feat6)



        if self.opt.isTrain and self.opt.novgg_featpair > 0:
            adaptive_feature_img_pair = self.adaptive_model_img(real_img, real_img)
            adaptive_feature_img_pair = util.feature_normalize(adaptive_feature_img_pair)
            coor_out['loss_novgg_featpair'] = nn.l1_loss(adp_feat_seg, adaptive_feature_img_pair) * self.opt.novgg_featpair

            if self.opt.mcl:
                feat_k, sample_ids = self.patch_sample(seg_feat6, 64, None)
                feat_q, _ = self.patch_sample(adaptive_feature_img_pair, 64, sample_ids)

                nceloss = self.nceloss(feat_k, feat_q)
                coor_out['nceloss'] = nceloss * self.opt.nce_w

        # 这里没用到
        if self.opt.use_coordconv:
            adp_feat_seg = self.addcoords(adp_feat_seg)
            adp_feat_img = self.addcoords(adp_feat_img)

        seg = nn.interpolate(seg_map, size=adp_feat_seg.size()[2:], mode='nearest')
        ref_seg = nn.interpolate(ref_seg_map, size=adp_feat_img.size()[2:], mode='nearest')

        if self.opt.maskmix:
            # 语义图特征 + 语义图 concat
            cont_features = self.layer(jittor.concat((adp_feat_seg, seg), 1))
            # 参考图特征 + 参考图的语义图 concat
            ref_features = self.layer(jittor.concat((adp_feat_img, ref_seg), 1))

        else:
            cont_features = self.layer(adp_feat_seg)
            ref_features = self.layer(adp_feat_img)

        dim_mean = 1 if self.opt.PONO_C else -1

        # feature branch
        # theta语义图
        theta, phi = cont_features, ref_features
        theta = self.theta(theta)
        theta = nn.unfold(theta, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        theta = util.mean_normalize(theta, dim_mean=dim_mean)
        theta_permute = theta.permute(0, 2, 1)

        # phi参考图
        phi = self.phi(phi)
        phi = nn.unfold(phi, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        phi = util.mean_normalize(phi, dim_mean=dim_mean)
        phi_permute = phi.permute(0, 2, 1)


        # weight branch
        # phi_w 参考图特征的 权重
        phi_w_feat = self.phi_w(ref_features).view(batch_size, self.cor_dim, -1)
        phi_w_feat = util.mean_normalize(phi_w_feat, dim_mean=dim_mean)
        phi_w_feat_pmt = phi_w_feat.permute(0, 2, 1)

        # theta_w 语义图特征的 权重
        theta_w_feat = self.theta_w(cont_features).view(batch_size, self.cor_dim, -1)
        theta_w_feat = util.mean_normalize(theta_w_feat, dim_mean=dim_mean)
        theta_w_feat_pmt = theta_w_feat.permute(0, 2, 1)

        _, N, D = theta_permute.shape

        # 语义图特征权重 和 参考图特征权重 -> 语义图权重
        theta_w = jittor.matmul(theta_w_feat_pmt, phi_w_feat.mean(-1, keepdims=True))
        theta_w = nn.softmax(theta_w.view(batch_size, N), dim=-1)

        # 参考图特征权重 和 语义图特征权重 -> 参考图权重
        phi_w = jittor.matmul(phi_w_feat_pmt, theta_w_feat.mean(-1, keepdims=True))
        phi_w = nn.softmax(phi_w.view(batch_size, N), dim=-1)
        
        # 语义图特征
        theta_w_ = theta_w.view(-1, 1, feat_height, feat_width).repeat(1, 3, 1, 1)
        # 参考图特征
        phi_w_ = phi_w.view(-1, 1, feat_height, feat_width).repeat(1, 3, 1, 1)
        
        # jittor默认bilinear
        coor_out['weight1'] = nn.interpolate(theta_w_, size=(im_height, im_width),mode='nearest')
        coor_out['weight2'] = nn.interpolate(phi_w_, size=(im_height, im_width),mode='nearest')




        # confidence branch
        # 语义图和参考图的特征，计算confidence
        phi_conf_feat = self.phi_conf(ref_features).view(batch_size, self.cor_dim, -1)
        phi_conf_feat = util.mean_normalize(phi_conf_feat, dim_mean=dim_mean)
        # phi_conf_feat_pmt = phi_conf_feat.permute(0, 2, 1)
        theta_conf_feat = self.theta_conf(cont_features).view(batch_size, self.cor_dim, -1)
        theta_conf_feat = util.mean_normalize(theta_conf_feat, dim_mean=dim_mean)
        theta_conf_feat_pmt = theta_conf_feat.permute(0, 2, 1)

        conf_map = jittor.matmul(theta_conf_feat_pmt, phi_conf_feat)
        #print(conf_map.shape) #[4,4096,4096,]
        conf_map = jittor.max(conf_map, -1, keepdims=True)
        #print(conf_map.shape) #[4096,1,]
        conf_map = (conf_map - conf_map.mean(dim=1, keepdims=True)).view(batch_size, 1, feat_height, feat_width)
        conf_map = jittor.sigmoid(conf_map*10.0)

        

        # phi_w = F.softmax(phi_w.view(batch_size, N) * 25.0, dim=-1)
        conf_map_ = conf_map.view(-1, 1, feat_height, feat_width).repeat(1, 3, 1, 1)
        # conf_map_ = conf_map_.view(-1, 1, 64, 64).repeat(1, 3, 1, 1)

        coor_out['conf_map'] = nn.interpolate(conf_map_, size=(im_height, im_width),mode='nearest')



        # OT matching branch
        try:
            F_, G_ = self.uot(theta_w, theta_permute, phi_w, phi_permute)
        except:
            print("check Var")
            print('theta_w_feat_pmt',theta_w_feat_pmt.min(), theta_w_feat_pmt.max(),'*'*20)
            print('phi_w_feat',phi_w_feat.min(), phi_w_feat.max(),'*'*20)
            print('ref_features',ref_features.min(), ref_features.max(),'*'*20)
            print('theta',theta_w.min(), theta_w.max(),'*'*20)
            print('ref_feat6',ref_feat6.min(), ref_feat6.max(),'*'*20)
            print('adp_feat_img',adp_feat_img.min(), adp_feat_img.max(),'*'*20)
            print('ref_seg',ref_seg.min(), ref_seg.max(),'*'*20)
            raise "inf"
            pass

        F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
        a_i, b_j = theta_w.view(-1, N, 1), phi_w.view(-1, N, 1)
        C_ij = 1 - jittor.matmul(theta_permute, phi)
        eps = self.blur ** self.p
        
        # softmax 得到 correspondence矩阵
        f = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)
        f_div_C = f / f.sum(-1).view(-1, N, 1)
        # f_div_C = F.softmax(f, dim=-1)
        # f = ((F_i + G_j - C_ij) / eps) * (a_i * b_j)
        # f_div_C = F.softmax(f*10000000, dim=-1)



        # feature transport branch
        # 参考图片 矩阵左乘 correspondence矩阵
        ref_ = nn.interpolate(ref_img, size=(feat_height, feat_width), mode='nearest')
        channel_ = ref_.shape[1]
        ref_ = ref_.view(batch_size, channel_, -1)
        ref_ = ref_.permute(0, 2, 1)
        f_div_C = f_div_C.float32()
        y1 = jittor.matmul(f_div_C, ref_)

        y_ = y1.permute(0, 2, 1)
        y_ = y_.view(batch_size, channel_, feat_height, feat_width)  # 2*3*44*44
        coor_out['warp_tmp'] = y_ if self.opt.warp_patch else self.upsampling(y_)

        ref_feat2 = nn.interpolate(ref_feat2, size=(feat_height, feat_width), mode='nearest')
        channel2 = ref_feat2.shape[1]
        ref_feat2 = ref_feat2.view(batch_size, channel2, -1).permute(0, 2, 1)
        y2 = jittor.matmul(f_div_C, ref_feat2)

        ref_feat3 = nn.interpolate(ref_feat3, size=(feat_height, feat_width), mode='nearest')
        channel3 = ref_feat3.shape[1]
        ref_feat3 = ref_feat3.view(batch_size, channel3, -1).permute(0, 2, 1)
        y3 = jittor.matmul(f_div_C, ref_feat3)


        ref_feat4 = nn.interpolate(ref_feat4, size=(feat_height, feat_width), mode='nearest')
        channel4 = ref_feat4.shape[1]
        ref_feat4 = ref_feat4.view(batch_size, channel4, -1).permute(0, 2, 1)
        y4 = jittor.matmul(f_div_C, ref_feat4)

        ref_feat5 = nn.interpolate(ref_feat5, size=(feat_height, feat_width), mode='nearest')
        channel5 = ref_feat5.shape[1]
        ref_feat5 = ref_feat5.view(batch_size, channel5, -1).permute(0, 2, 1)
        y5 = jittor.matmul(f_div_C, ref_feat5)

        # 不同尺度的wraping
        y1 = y1.permute(0, 2, 1).view(batch_size, channel_, feat_height, feat_width)
        y2 = y2.permute(0, 2, 1).view(batch_size, channel2, feat_height, feat_width)
        y3 = y3.permute(0, 2, 1).view(batch_size, channel3, feat_height, feat_width)
        y4 = y4.permute(0, 2, 1).view(batch_size, channel4, feat_height, feat_width)
        y5 = y5.permute(0, 2, 1).view(batch_size, channel5, feat_height, feat_width)

        coor_out['warp_out'] = [seg_map, seg_feat2, seg_feat3, seg_feat4, seg_feat5, y1, y2, y3, y4, y5, conf_map]


        if self.opt.warp_mask_losstype == 'direct' or self.opt.show_warpmask:
            ref_seg = nn.interpolate(ref_seg_map, scale_factor= 1/self.down, mode='nearest')
            channel = ref_seg.shape[1]
            ref_seg = ref_seg.view(batch_size, channel, -1)
            ref_seg = ref_seg.permute(0, 2, 1)
            warp_mask = jittor.matmul(f_div_C, ref_seg)  # 2*1936*channel
            warp_mask = warp_mask.permute(0, 2, 1)
            coor_out['warp_mask'] = warp_mask.view(batch_size, channel, feat_height, feat_width)  # 2*3*44*44
        elif self.opt.warp_mask_losstype == 'cycle':
            # f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)
            f_WTA_v = f.transpose(1, 2)
            f_div_C_v = f_WTA_v / f_WTA_v.sum(-1).view(-1, N, 1)

            seg = nn.interpolate(seg_map, scale_factor=1 / self.down, mode='nearest')
            channel = seg.shape[1]
            seg = seg.view(batch_size, channel, -1)
            seg = seg.permute(0, 2, 1)
            warp_mask_to_ref = jittor.matmul(f_div_C_v, seg)  # 2*1936*channel
            warp_mask = jittor.matmul(f_div_C, warp_mask_to_ref)  # 2*1936*channel
            warp_mask = warp_mask.permute(0, 2, 1)
            coor_out['warp_mask'] = warp_mask.view(batch_size, channel, feat_height, feat_width)  # 2*3*44*44
        else:
            warp_mask = None

        if self.opt.warp_cycle_w > 0:
            if self.opt.correspondence == 'ot':
                f_WTA_v = f.transpose(1, 2)
                f_div_C_v = f_WTA_v / f_WTA_v.sum(-1).view(-1, N, 1)
            else:
                f_div_C_v = nn.softmax(f.transpose(1, 2), dim=-1)


            if self.opt.warp_patch:
                y_ = nn.unfold(y_, self.down, stride=self.down)
                warp_cycle = jittor.matmul(f_div_C_v, y_)
                warp_cycle = warp_cycle.permute(0, 2, 1)
                warp_cycle = nn.fold(warp_cycle, 256, self.down, stride=self.down)
                coor_out['warp_cycle'] = warp_cycle
            else:
                channel = y_.shape[1]
                y_ = y_.view(batch_size, channel, -1).permute(0, 2, 1)
                warp_cycle = jittor.matmul(f_div_C_v, y_).permute(0, 2, 1)

                coor_out['warp_cycle'] = warp_cycle.view(batch_size, channel, feat_height, feat_width)
                if self.opt.two_cycle:
                    real_img = nn.avg_pool2d(real_img, self.down)
                    real_img = real_img.view(batch_size, channel, -1)
                    real_img = real_img.permute(0, 2, 1)
                    warp_i2r = jittor.matmul(f_div_C_v, real_img).permute(0, 2, 1)  #warp input to ref
                    warp_i2r = warp_i2r.view(batch_size, channel, feat_height, feat_width)
                    warp_i2r2i = jittor.matmul(f_div_C, warp_i2r.view(batch_size, channel, -1).permute(0, 2, 1))
                    coor_out['warp_i2r'] = warp_i2r
                    coor_out['warp_i2r2i'] = warp_i2r2i.permute(0, 2, 1).view(batch_size, channel, feat_height, feat_width)

        return coor_out

    def addcoords(self, x):
        bs, _, h, w = x.shape

        xx_ones = jittor.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_ones.stop_grad()
        xx_range = jittor.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_range = xx_range.stop_grad()
        xx_channel = jittor.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = jittor.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_ones.stop_grad()
        yy_range = jittor.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_range = yy_range.stop_grad()
        yy_channel = jittor.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = jittor.sqrt(jittor.pow(xx_channel, 2) + jittor.pow(yy_channel, 2))

        concat = jittor.concat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')

    # for training
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

    # for discriminators
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

    parser.add_argument('--which_perceptual', type=str, default='5_2', help='relu5_2 or relu4_2')
    parser.add_argument('--weight_perceptual', type=float, default=0.01)
    parser.add_argument('--weight_mask', type=float, default=0.0, help='weight of warped mask loss, used in direct/cycle')
    parser.add_argument('--real_reference_probability', type=float, default=0.7, help='self-supervised training probability')
    parser.add_argument('--hard_reference_probability', type=float, default=0.2, help='hard reference training probability')
    parser.add_argument('--weight_gan', type=float, default=10.0, help='weight of all loss in stage1')
    parser.add_argument('--novgg_featpair', type=float, default=10.0, help='in no vgg setting, use pair feat loss in domain adaptation')
    parser.add_argument('--D_cam', type=float, default=0.0, help='weight of CAM loss in D')
    parser.add_argument('--warp_self_w', type=float, default=0.0, help='push warp self to ref')
    parser.add_argument('--fm_ratio', type=float, default=0.1, help='vgg fm loss weight comp with ctx loss')
    parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
    parser.add_argument('--ctx_w', type=float, default=1.0, help='ctx loss weight')
    parser.add_argument('--mask_epoch', type=int, default=-1, help='useful when noise_for_mask is true, first train mask_epoch with mask, the rest epoch with noise')

    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    parser.add_argument('--blur', type=float, default=0.005, help='blur in OT')
    parser.add_argument('--correspondence', type=str, default='ot', help='ot, euc')
    parser.add_argument('--ot_weight', action='store_true', help='use euc distance as weight of ot')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    # for setting inputs
    parser.add_argument('--dataroot', type=str, default='/mnt/blob/Dataset/ADEChallengeData2016/images')
    parser.add_argument('--dataset_mode', type=str, default='ade20k')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
    parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

    # for displays
    parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

    # for generator
    parser.add_argument('--netG', type=str, default='seace', help='selects model to use for netG (pix2pixhd | spade | seace)')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--z_dim', type=int, default=256,
                        help="dimension of the latent z vector")

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
    parser.add_argument('--semantic_nc', type=int, default=29, help='correspondence matrix match kernel size')

    # net corr
    parser.set_defaults(norm_G='spectralspadebatch3x3')
    parser.add_argument('--num_upsampling_layers',
        choices=('normal', 'more', 'most'), default='normal',
        help="If 'more', adds upsampling layer between the two middle resnet blocks")
    
    parser.add_argument('--isTrain', action='store_true', help='useful in deepfashion')


    opt = parser.parse_args()
    netCorr = NoVGGCorrespondence(opt)
    ref_image = jittor.randn((4,3,256,256))
    real_image = jittor.randn((4,3,256,256))
    input_semantics = jittor.randn((4,29,256,256))
    ref_semantics = jittor.randn((4,29,256,256))
    coor_out = netCorr(ref_image, real_image, input_semantics, ref_semantics)
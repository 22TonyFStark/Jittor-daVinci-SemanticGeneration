import jittor.nn as nn
import models.networks as networks
import util.util as util
import jittor
from models.networks.ContextualLoss import ContextualLoss_forward
from models.piq.dists import DISTS





def weighted_l1_loss(input, target, weights):
    out = jittor.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def mse_loss(input, target=0):
    return jittor.mean((input - target)**2)



class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.alpha = 1

        self.net = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            
            self.vggnet_fix = networks.correspondence.VGG19_feature_color_jittor(vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.eval()

            for param in self.vggnet_fix.parameters():
                param.stop_grad()
            
            
            self.contextual_forward_loss = ContextualLoss_forward(opt)
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.MSE_loss = nn.MSELoss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2
            if opt.dists_w != 0:
                self.DISTSLoss = DISTS()

    def execute(self, data, mode, GforD=None, alpha=1):

        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.preprocess_input(data.copy(), )
        # print(input_label.shape, input_semantics.shape, real_image.shape, ref_image.shape, ref_label.shape, ref_semantics.shape)
        # [1,1,256,256,] [1,29,256,256,] [1,3,256,256,] [1,3,256,256,] [1,1,256,256,] [1,29,256,256,]
        self.alpha = alpha
        generated_out = {}
        if mode == 'generator':
            g_loss, generated_out = self.compute_generator_loss(input_label,
                input_semantics, real_image, ref_label, ref_semantics, ref_image, self_ref)
            
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['warp_mask'] = None if 'warp_mask' not in generated_out else generated_out['warp_mask']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            out['warp_i2r'] = None if 'warp_i2r' not in generated_out else generated_out['warp_i2r']
            out['warp_i2r2i'] = None if 'warp_i2r2i' not in generated_out else generated_out['warp_i2r2i']
            out['warp_tmp'] = None if 'warp_tmp' not in generated_out else generated_out['warp_tmp']
            out['weight1'] = None if 'weight1' not in generated_out else generated_out['weight1']
            out['weight2'] = None if 'weight2' not in generated_out else generated_out['weight2']
            out['conf_map'] = None if 'conf_map' not in generated_out else generated_out['conf_map']


            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, ref_image, GforD, label=input_label)
            return d_loss
        elif mode == 'inference':
            out = {}
            with jittor.no_grad():
                out = self.inference(input_semantics, 
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")


    def create_optimizers(self, opt):
        G_params, D_params = list(), list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr * 0.5}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr * 0.5}]

        if opt.isTrain:
            D_params += list(self.net['netD'].parameters())
            if opt.weight_domainC > 0 and opt.domain_rela:
                D_params += list(self.net['netDomainClassifier'].parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jittor.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = jittor.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)
        if self.opt.weight_domainC > 0 and self.opt.domain_rela: 
            util.save_network(self.net['netDomainClassifier'], 'DomainClassifier', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netD'] = networks.define_D(opt) if opt.isTrain else None
        net['netCorr'] = networks.define_Corr(opt)
        net['netDomainClassifier'] = networks.define_DomainClassifier(opt) if opt.weight_domainC > 0 and opt.domain_rela else None
        
        # pretrain
        """
        pretrain = True
        if pretrain:
            print("loaded pretrained weights")
            net['netG'].load_state_dict(jittor.load("checkpoints/UNITE_eqlrsn_nce_dists_x256/85_net_G.pkl"))
            if opt.isTrain:
                net['netD'].load_state_dict(jittor.load("checkpoints/UNITE_eqlrsn_nce_dists_x256/85_net_D.pkl"))
            net['netCorr'].load_state_dict(jittor.load("checkpoints/UNITE_eqlrsn_nce_dists_x256/85_net_Corr.pkl"))
        """
        # train
        
        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
            #if not self.opt.skip_corr:
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
            if opt.weight_domainC > 0 and opt.domain_rela:
                net['netDomainClassifier'] = util.load_network(net['netDomainClassifier'], 'DomainClassifier', opt.which_epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                net['netG'] = util.load_network(net['netG'], 'G_ema', opt.which_epoch, opt)
                
                net['netCorr'] = util.load_network(net['netCorr'], 'netCorr_ema', opt.which_epoch, opt)
        
        return net
        #return netG_stage1, netD_stage1, netG, netD, netE, netCorr

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        if self.opt.dataset_mode == 'celebahq':
            glasses = data['label'][:,1::2,:,:].long()
            data['label'] = data['label'][:,::2,:,:]
            glasses_ref = data['label_ref'][:,1::2,:,:].long()
            data['label_ref'] = data['label_ref'][:,::2,:,:]
            if self.use_gpu():
                glasses = glasses
                glasses_ref = glasses_ref
        elif self.opt.dataset_mode == 'celebahqedge':
            input_semantics = data['label'].clone()
            data['label'] = data['label'][:,:1,:,:]
            ref_semantics = data['label_ref'].clone()
            data['label_ref'] = data['label_ref'][:,:1,:,:]
        elif self.opt.dataset_mode == 'deepfashion':
            input_semantics = data['label'].clone()
            data['label'] = data['label'][:,:3,:,:]
            ref_semantics = data['label_ref'].clone()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        elif self.opt.dataset_mode == 'ade20klayout':
            input_semantics = data['label'][:, 3:,:,:].clone()
            data['label'] = data['label'][:, :3,:,:]
            ref_semantics = data['label_ref'][:, 3:,:,:].clone()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        elif self.opt.dataset_mode == 'cocolayout':
            input_semantics = data['label'][:, 3:,:,:].clone()
            data['label'] = data['label'][:, :3,:,:]
            ref_semantics = data['label_ref'][:, 3:,:,:].clone()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        if self.opt.dataset_mode != 'deepfashion':
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label']
            data['image'] = data['image']
            data['ref'] = data['ref']
            data['label_ref'] = data['label_ref']
            if self.opt.dataset_mode != 'deepfashion':
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref']

        # create one-hot label map
        # if self.opt.dataset_mode != 'celebahqedge' and self.opt.dataset_mode != 'deepfashion':
        if self.opt.dataset_mode == 'ade20k' or self.opt.dataset_mode == 'coco' or self.opt.dataset_mode == 'scene':
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = jittor.zeros((bs, nc, h, w))
            input_label.stop_grad()
            src = jittor.ones((bs, nc, h, w))
            src.stop_grad()
            input_semantics = input_label.scatter_(1, label_map, src)
        
            label_map = data['label_ref']
            label_ref = jittor.zeros((bs, nc, h, w))
            label_ref.stop_grad()
            src = jittor.ones((bs, nc, h, w))
            src.stop_grad()
            ref_semantics = label_ref.scatter_(1, label_map, src)

        if self.opt.dataset_mode == 'celebahq':
            assert input_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            input_semantics[:,-3:-2,:,:] = glasses
            assert ref_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            ref_semantics[:,-3:-2,:,:] = glasses_ref

        # print (input_semantics.min(), input_semantics.max())
        # print (1/0)

        h_resize = int(data['image'].shape[-1] / self.opt.aspect_ratio)
        w_resize = data['image'].shape[-1]
        input_semantics = nn.resize(input_semantics, [h_resize, w_resize],mode="nearest")
        data['image'] = nn.resize(data['image'], [h_resize, w_resize],mode="bicubic")
        ref_semantics = nn.resize(ref_semantics, [h_resize, w_resize],mode="nearest")
        data['ref'] = nn.resize(data['ref'], [h_resize, w_resize],mode="bicubic")

        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = jittor.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = jittor.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = jittor.mean(self.contextual_forward_loss(nn.avg_pool2d(source[-3], 2), nn.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = jittor.mean(self.contextual_forward_loss(nn.avg_pool2d(source[-4], 4), nn.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None, ref_image=None, self_ref=None):
        G_losses = {}
        generate_out = self.generate_fake(
            input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)

        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']

        if self.opt.warp_cycle_w > 0:
            if not self.opt.warp_patch:
                ref = nn.avg_pool2d(ref_image, self.opt.warp_stride)
            else:
                ref = ref_image

            G_losses['G_warp_cycle'] = nn.l1_loss(generate_out['warp_cycle'], ref) * self.opt.warp_cycle_w
            if self.opt.two_cycle:
                real = nn.avg_pool2d(real_image, self.opt.warp_stride)
                G_losses['G_warp_cycle'] += nn.l1_loss(generate_out['warp_i2r2i'], real) * self.opt.warp_cycle_w
                
        if self.opt.warp_self_w > 0:
            sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            G_losses['G_warp_self'] = jittor.mean(nn.l1_loss(generate_out['warp_tmp'], real_image) * sample_weights) * self.opt.warp_self_w

        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, generate_out['fake_image'], real_image, ref_image)
        #print('输入GAN LOSS：',pred_fake[0][0].flatten()[:20:2], 'weight:',self.opt.weight_gan)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False) * self.opt.weight_gan
        #print('输出GAN Loss',G_losses['GAN'])
        
        if self.opt.dists_w != 0:
            fake_image_norm = (generate_out['fake_image']+1)*0.5
            real_image_norm = (real_image+1)*0.5
            G_losses['DISTS'] = self.DISTSLoss(fake_image_norm, real_image_norm) * self.opt.dists_w

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = jittor.zeros(1)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #print(self_ref.shape)
        #print('sample_weights',sample_weights)
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(), sample_weights)

        G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio
        
        feat_loss = mse_loss(fake_features[self.perceptual_layer], generate_out['real_features'][self.perceptual_layer].detach())

        G_losses['perc'] = feat_loss * self.opt.weight_perceptual

        G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w

        if self.opt.warp_mask_losstype != 'none':
            ref_label = nn.interpolate(ref_label.float(), scale_factor=1/self.opt.warp_stride, mode='nearest').long().squeeze(1)
            gt_label = nn.interpolate(input_label.float(), scale_factor=1/self.opt.warp_stride, mode='nearest').long().squeeze(1)
            weights = []
            for i in range(ref_label.shape[0]):
                ref_label_uniq = jittor.misc.unique(ref_label[i])
                gt_label_uniq = jittor.misc.unique(gt_label[i])
                zero_label = [it for it in gt_label_uniq if it not in ref_label_uniq]
                weight = jittor.ones_like(gt_label[i]).float()
                weight.stop_grad()
                for j in zero_label:
                    weight[gt_label[i] == j] = 0
                weight[gt_label[i] == 0] = 0 #no loss from unknown class
                weights.append(weight.unsqueeze(0))
            weights = jittor.concat(weights, dim=0)

            # print (generate_out['warp_mask'].min(), generate_out['warp_mask'].max())
            # print (input_semantics.min(), input_semantics.max())

            if self.opt.dataset_mode == 'ade20klayout' or 'cocolayout':
                gt_label = nn.interpolate(input_semantics.float(), scale_factor=1 / self.opt.warp_stride,
                                         mode='nearest').long()#.squeeze(1)


                # G_losses['mask'] = util.mse_loss(generate_out['warp_mask'].float(), gt_label.float()) * 200
                G_losses['mask'] = nn.l1_loss(generate_out['warp_mask'].float(), gt_label.float()) * 500
            else:
                
                G_losses['mask'] = (nn.nll_loss(jittor.log(generate_out['warp_mask'] + 1e-10), gt_label, reduce =False)
                                * weights).sum() / (weights.sum() + 1e-5) * self.opt.weight_mask
                

            #print('warp mask',generate_out['warp_mask'].flatten()[::100])
            #print('gt_label',gt_label.flatten()[::100])
            #print('weight mask', self.opt.weight_mask)
            #print('loss', G_losses['mask'])
            #raise "check mask loss"

            # print (G_losses['mask'])
            # print (1/0)
        #self.fake_image = fake_image
        #print('G LOSS')
        #for loss_name in G_losses.keys():
        #    print(loss_name, G_losses[loss_name])
        #raise "check G loss"

        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, ref_image, GforD, label=None):
        D_losses = {}
        with jittor.no_grad():
            #fake_image, _, _, _, _ = self.generate_fake(input_semantics, real_image, VGG_feat=False)
            fake_image = GforD['fake_image'].detach()
            #fake_image.requires_grad_()

        pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit = self.discriminate(
            input_semantics, fake_image, real_image, ref_image)

        #print('pred fake and real')

        #print(pred_fake[0][-1].flatten()[::100])
        #print(pred_real[0][-1].flatten()[::100])
        
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.weight_gan
        #print('D LOSS')
        #for loss_name in D_losses.keys():
        #    print(loss_name, D_losses[loss_name])
        #raise "check loss"
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.net['netE'](real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1 = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        
        coor_out = self.net['netCorr'](ref_image, real_image, input_semantics, ref_semantics)

        generate_out['ref_features'] = [ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1]
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

        generate_out['fake_image'] = self.net['netG'](warp_out=coor_out['warp_out'])

        generate_out = {**generate_out, **coor_out}
        return generate_out

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None):

        #print(ref_image.shape) #[16,3,192,256,]

        #input_semantics = nn.resize(input_semantics, [im_height, im_width],mode="nearest")
        #ref_image = nn.resize(ref_image, [im_height, im_width],mode="bicubic")
        #ref_semantics = nn.resize(ref_semantics, [im_height, im_width],mode="nearest")

        generate_out = {}
        coor_out = self.net['netCorr'](ref_image, None, input_semantics, ref_semantics)
        # atten_map = coor_out['atten_map']
        generate_out['fake_image'] = self.net['netG'](warp_out=coor_out['warp_out'])

        generate_out = {**generate_out, **coor_out}
        return generate_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, ref_image):
        fake_concat = jittor.concat([input_semantics, fake_image], dim=1)
        real_concat = jittor.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.

        fake_and_real = jittor.concat([fake_concat, real_concat], dim=0)


        #print('D input',fake_and_real.flatten()[0::15000], fake_and_real.sum())
        
        seg = None


        discriminator_out, seg, cam_logit = self.net['netD'](fake_and_real)
        


        #raise "check D"
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        fake_cam_logit, real_cam_logit = None, None
        if self.opt.D_cam > 0:
            fake_cam_logit = jittor.concat([it[:it.shape[0]//2] for it in cam_logit], dim=1)
            real_cam_logit = jittor.concat([it[it.shape[0]//2:] for it in cam_logit], dim=1)
        #fake_cam_logit, real_cam_logit = self.divide_pred(cam_logit)

        return pred_fake, pred_real, seg, fake_cam_logit, real_cam_logit

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = jittor.zeros(t.size())
        edge.stop_grad()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jittor.exp(0.5 * logvar)
        eps = jittor.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = nn.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = nn.cross_entropy(real_seg[0][0], gt)

        down_gt = nn.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().long()
        fake_seg_loss_down = nn.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = nn.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss



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
    parser.add_argument('--isTrain', action='store_true', help='useful in deepfashion')
    parser.add_argument('--which_epoch', type=str, default='latest', help='useful in deepfashion')

    opt = parser.parse_args()

    opt.semantic_nc = 29
    opt.num_D = 2
    opt.netD_subarch = 'n_layer'
    

    G_test = Pix2PixModel(opt)
    print(G_test)
    #inp = jittor.randn((4,))

if __name__ == '__main__':
    test()
    pass
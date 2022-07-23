# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from models.pix2pix_model import Pix2PixModel
import jittor

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, resume_epoch=0):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr

        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None, None, None

    def run_generator_one_step(self, data, alpha=1):

        g_losses, out = self.pix2pix_model(data, mode='generator', alpha=alpha)

        g_loss = sum(g_losses.values()).mean()
        
        """
        For example, if your code looks like this::
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        It can be changed to this::
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

        Or more concise::

            optimizer.step(loss)
        """
        
        """ grad clip """
        self.optimizer_G.zero_grad()
        self.optimizer_G.backward(g_loss)
        self.optimizer_G.clip_grad_norm(1, 2)
        self.optimizer_G.step()
        

        #self.optimizer_G.step(g_loss)
        
        
        self.g_losses = g_losses
        self.out = out

    def run_discriminator_one_step(self, data):

        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']
        d_losses = self.pix2pix_model(data, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()

        #self.optimizer_D.step(d_loss)
        """ grad clip """
        self.optimizer_D.zero_grad()
        self.optimizer_D.backward(d_loss)
        self.optimizer_D.clip_grad_norm(1, 2)
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    @jittor.single_process_scope()
    def save(self, epoch):
        self.pix2pix_model.save(epoch)

        if epoch == 'latest':
            jittor.save({'G': self.optimizer_G.state_dict(),
                        'D': self.optimizer_D.state_dict(),
                        'lr':  self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pkl'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def update_fixed_params(self):
        for param in self.pix2pix_model.net['netCorr'].parameters():
            param.requires_grad = True
        G_params = [{'params': self.pix2pix_model.net['netG'].parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model.net['netCorr'].parameters(), 'lr': self.opt.lr*0.5}]
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr = self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr = self.opt.lr / 2

        self.optimizer_G = jittor.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import jittor.misc as vutils
import sys
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from trainers.pix2pix_trainer import Pix2PixTrainer
import jittor
import jittor.nn as nn

jittor.cudnn.set_max_workspace_ratio(0.0)
jittor.flags.use_cuda = 1

# parse options
opt = TrainOptions().parse()

# print options to help debugging
if jittor.rank == 0:
    print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)
#dataloader.dataset[11]

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)


# save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), opt.name)
for epoch in iter_counter.training_epochs():
    opt.epoch = epoch
    
    iter_counter.record_epoch_start(epoch)

    if jittor.rank == 0:
        if not opt.maskmix:
            print('inject nothing')
        elif opt.maskmix and opt.noise_for_mask and epoch > opt.mask_epoch:
            print('inject noise')
        else:
            print('inject mask')
        print('real_reference_probability is :{}'.format(dataloader.real_reference_probability))
        print('hard_reference_probability is :{}'.format(dataloader.hard_reference_probability))

    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):

        iter_counter.record_one_iteration()

        #use for Domain adaptation loss
        p = min(float(i + (epoch - 1) * len_dataloader) / 50 / len_dataloader, 1)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # Training
        # try:
        if 1:
            # train
            try:
                if i % opt.D_steps_per_G == 0:
                    trainer.run_generator_one_step(data_i, alpha=alpha)
                trainer.run_discriminator_one_step(data_i)
            except:
                continue

            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                try:
                    print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                                    losses, iter_counter.time_per_iter)
                except OSError as err:
                    print(err)

            if iter_counter.needs_displaying():
                imgs_num = data_i['label'].shape[0]
                if opt.dataset_mode == 'celebahq':
                    data_i['label'] = data_i['label'][:,::2,:,:]
                elif opt.dataset_mode == 'celebahqedge':
                    data_i['label'] = data_i['label'][:,:1,:,:]
                elif opt.dataset_mode == 'deepfashion':
                    data_i['label'] = data_i['label'][:,:3,:,:]
                elif opt.dataset_mode == 'ade20klayout':
                    data_i['label'] = (data_i['label'][:,:3,:,:] -128) / 128
                elif opt.dataset_mode == 'cocolayout':
                    data_i['label'] = (data_i['label'][:,:3,:,:] -128) / 128
                if data_i['label'].shape[1] == 3:
                    label = data_i['label']
                else:
                    label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()

                label = nn.resize(label, (opt.im_height, opt.im_width), mode='nearest')
                data_i['ref'] = nn.resize(data_i['ref'], (opt.im_height, opt.im_width), mode='bicubic')
                data_i['image'] = nn.resize(data_i['image'], (opt.im_height, opt.im_width), mode='bicubic')

                imgs = jittor.concat((label, trainer.out['weight1']*255, trainer.out['weight2']*255,
                                    data_i['ref'], trainer.out['warp_tmp'],
                                    trainer.get_latest_generated().data, data_i['image']), 0)

                im_sv_path = opt.checkpoints_dir.split('checkpoints')[0] + 'summary/'  +opt.name
                if jittor.rank == 0:
                    if not os.path.exists(im_sv_path):
                        os.makedirs(im_sv_path)
                try:
                    vutils.save_image(imgs, im_sv_path + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_' +'gpu'+str(jittor.rank)+'.png',
                            nrow=imgs_num, padding=0, normalize=True)
                except OSError as err:
                    print(err)

                # print(trainer.out['weight1'].min(), trainer.out['weight1'].max())
                # print(trainer.out['conf_map'].min(), trainer.out['conf_map'].max())
                # print(trainer.out['conf_map'].shape)
                # print(1 / 0)

        # except Exception as e:
        #     print(e)
        # trainer.save('latest_')
        # continue
        
        if iter_counter.needs_saving():
            try:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)
        


    trainer.update_learning_rate(epoch)

    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        try:
            print('saving the model at the end of epoch %d, iters %d' %
            (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

    jittor.sync_all()
    jittor.gc()

print('Training was successfully finished.')
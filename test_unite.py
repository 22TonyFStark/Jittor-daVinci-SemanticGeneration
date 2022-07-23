# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import jittor
import jittor.misc as vutils
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import matplotlib.pyplot as plt
jittor.cudnn.set_max_workspace_ratio(0.0)
jittor.flags.use_cuda = 1


opt = TestOptions().parse()
   
dataloader = data.create_dataloader(opt)


model = Pix2PixModel(opt)
model.eval()

# save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')
save_root = opt.checkpoints_dir.split('checkpoints')[0] + 'results/' + opt.name + str(opt.which_epoch)+'/'



# test
for i, data_i in enumerate(dataloader):
    if jittor.rank == 0:
        print('{} / {}'.format(i, len(dataloader)))
    # if i * opt.batchSize >= 4993:
    # if i * opt.batchSize >= 400:
    #     break
    imgs_num = data_i['label'].shape[0]
    # out = model(data_i, mode='inference')
    out = model(data_i, mode='inference')

    if jittor.rank == 0:
        if not os.path.exists(save_root + '/pre'):
            os.makedirs(save_root + '/pre')
            os.makedirs(save_root + '/gt')

    pre = out['fake_image'].data
    gt = data_i['image']
    ref = data_i['ref']
    label = data_i['label'][:, :1, :, :] + 0.5
    path = data_i['path']
    warp = out['warp_tmp'].data

    batch_size = pre.shape[0]

    for j in range(batch_size):
        pre_ = pre[j]
        gt_ = gt[j]
        ref_ = ref[j]
        label_ = label
        warp_ = warp[j]
        name = path[j].split('/')[-1]

        pre_ = (pre_ + 1) / 2
        vutils.save_image(jittor.array(pre_), save_root + '/pre/' +name.replace('jpg', 'png'),
                nrow=imgs_num, padding=0, normalize=False)

        gt_ = (gt_ + 1) / 2
        vutils.save_image(jittor.array(gt_), save_root + '/gt/' +name.replace('jpg', 'png'),
                          nrow=imgs_num, padding=0, normalize=False)

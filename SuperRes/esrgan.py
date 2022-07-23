

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import math
import itertools
import sys
import cv2

from models import *
from datasets import *

import jittor as jt

jt.flags.use_cuda = 1

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="train_img", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=384, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=2500, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=8, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)


def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=1)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_content = nn.L1Loss()
criterion_pixel = nn.L1Loss()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(jt.load("saved_models/generator_%d.pkl" % opt.epoch))

# Optimizers
optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

train_dataset = ImageDataset("/home/user/duzongwei/Projects/JTGAN/SPADE/datasets/train/train_img", hr_shape=hr_shape)

dataloader = train_dataset.set_attrs(
    batch_size=opt.batch_size, 
    shuffle=True, 
    num_workers=opt.n_cpu
)
# ----------
#  Training
# ----------
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = imgs["lr"]
        imgs_hr = imgs["hr"]

        # Adversarial ground truths
        valid = jt.array(np.ones((imgs_lr.size(0), *discriminator.output_shape))).float32().stop_grad()
        fake = jt.array(np.zeros((imgs_lr.size(0), *discriminator.output_shape))).float32().stop_grad()

        # ------------------
        #  Train Generators
        # ------------------


        # Generate a high resolution image from low resolution input
        # print(imgs_lr.shape)
        gen_hr = generator(imgs_lr)

        # print(gen_hr.shape, imgs_hr.shape)
        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            optimizer_G.step(loss_pixel)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.data[0])
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).stop_grad()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).stop_grad()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        optimizer_G.step(loss_G)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.stop_grad())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.step(loss_D)

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.data[0],
                loss_G.data[0],
                loss_content.data[0],
                loss_GAN.data[0],
                loss_pixel.data[0],
            )
        )

        # just test, note at training
        # print(imgs_lr.shape)
        # imgs_lr = nn.interpolate(imgs_lr, scale_factor=2)
        # print(imgs_lr.shape, gen_hr.shape)
        # img_grid = denormalize(jt.contrib.concat((imgs_lr, gen_hr), -1))
        # print(img_grid.shape)
        # save_image(img_grid.numpy(), "images/training/%d.png" % batches_done, nrow=1)
        
        # jt.save(generator.state_dict(), "saved_models/generator_%d.pkl" % epoch)
        

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            # imgs_lr = nn.Upsample(2)(imgs_lr)
            imgs_lr = nn.interpolate(imgs_lr, scale_factor=2)
            img_grid = denormalize(jt.contrib.concat((imgs_lr, gen_hr), -1))
            save_image(img_grid.numpy(), "images/training/%d.png" % batches_done, nrow=1)
            
        if batches_done % opt.checkpoint_interval == 0 and jt.rank == 0:
                # Save model checkpoints
            jt.save(generator.state_dict(), "saved_models/generator_%d.pkl" % epoch)
            jt.save(discriminator.state_dict(), "saved_models/discriminator_%d.pkl" %epoch)

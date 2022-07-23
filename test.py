import os
import jittor
import jittor.misc as vutils
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import matplotlib.pyplot as plt
from SuperRes.models import GeneratorRRDB
from SuperRes.datasets import denormalize, mean, std
from jittor import transform as transforms
from PIL import Image
import numpy as np
from jittor.misc import save_image
from tqdm import tqdm
jittor.cudnn.set_max_workspace_ratio(0.0)
jittor.flags.use_cuda = 1
opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)
# stage1 generation
model = Pix2PixModel(opt)
model.eval()
save_root = opt.output_path
for i, data_i in enumerate(dataloader):
    if jittor.rank == 0:
        print('{} / {}'.format(i, len(dataloader)))
    imgs_num = data_i['label'].shape[0]
    out = model(data_i, mode='inference')
    pre = out['fake_image'].data#.cpu()
    gt = data_i['image']#.cpu()
    ref = data_i['ref']#.cpu()
    label = data_i['label'][:, :1, :, :] + 0.5
    path = data_i['path']
    warp = out['warp_tmp'].data#.cpu()
    batch_size = pre.shape[0]
    for j in range(batch_size):
        pre_ = pre[j]
        gt_ = gt[j]
        ref_ = ref[j]
        label_ = label
        warp_ = warp[j]
        name = path[j].split('/')[-1]
        pre_ = (pre_ + 1) / 2
        vutils.save_image(jittor.array(pre_), save_root + '/' +name.replace('jpg', 'png'),
                nrow=imgs_num, padding=0, normalize=False)
# stage2 superresolution
generator = GeneratorRRDB(3, filters=64, num_res_blocks=8, num_upsample=1)
generator.load_state_dict(jittor.load("./weights/EsrganG.pkl"))
generator.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.ImageNormalize(mean, std)])
image_paths = [os.path.join(save_root, p) for p in os.listdir(save_root)]
print("super-resolution:")
for image_p in image_paths:
    print('processing', image_p)
    image_tensor = jittor.Var((transform(Image.open(image_p)))).unsqueeze(0)
    assert image_tensor.shape[-2] == 256 or image_tensor.shape[-2] == 192
    image_tensor = jittor.nn.interpolate(image_tensor, size=(192,256), mode="bicubic")
    with jittor.no_grad():
        sr_image = denormalize(generator(image_tensor)).squeeze(0)
    fn = image_p.split("/")[-1]
    fn.replace("jpg","png")
    save_image(sr_image, image_p, nrow=1)
# convert image format
import glob
imgs = glob.glob(f"{save_root}/*.png")
print(f'tatally generate {len(imgs)} image(s)')

for img in imgs:
    code = f"convert {img} {img[:-4]}.jpg"
    print(code)
    os.system(code)
for img in imgs:
    os.remove(img)
print("done!")

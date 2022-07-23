from models import GeneratorRRDB
from datasets import denormalize, mean, std
import jittor
import argparse
import os
from jittor import transform as transforms
from PIL import Image
import numpy as np
from jittor.misc import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
#parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=8, help="Number of residual blocks in G")
parser.add_argument("--local_rank", type=int)
parser.add_argument('--rescale', action='store_true', help='rescale')
parser.add_argument("--savename", default="pth_test", type=str)

opt = parser.parse_args()
print(opt)

os.makedirs(f"images/{opt.savename}", exist_ok=True)


# CUDA
jittor.cudnn.set_max_workspace_ratio(0.0)
jittor.flags.use_cuda = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '3'



# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=1)


generator.load_state_dict(jittor.load("/home/qingzhongfei/A_scene/unite_jittor/weights/EsrganG.pkl"))


generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.ImageNormalize(mean, std)])


image_paths = [p for p in os.listdir(opt.image_path) if "png" in p]
for image_p in image_paths:
    print("processing", image_p)
    image_p = os.path.join(opt.image_path,image_p)

    image_tensor = jittor.Var((transform(Image.open(image_p)))).unsqueeze(0)
    assert image_tensor.shape[-2] == 256 or image_tensor.shape[-2] == 192
    
    # 先rescale再超分
    if opt.rescale:
        image_tensor = jittor.nn.interpolate(image_tensor, size=(192,256), mode="bicubic")
        
 
    # Upsample image
    with jittor.no_grad():
        sr_image = denormalize(generator(image_tensor)).squeeze(0)
        # sr_image = denormalize(generator(image_tensor))

    # Save image
    fn = image_p.split("/")[-1]
    fn.replace("jpg","png")
    
    # 先超分再rescale，弃用
    # if opt.rescale:
    #     sr_image = jittor.nn.interpolate(sr_image.unsqueeze(0), size=(384, 512), mode="bicubic").squeeze(0)

    print(sr_image.shape)
    save_image(sr_image, f"images/{opt.savename}/{fn}", nrow=1)
print("done!")

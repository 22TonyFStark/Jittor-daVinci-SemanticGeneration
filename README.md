# Jittor 草图生成风景比赛 UNITE + SESAME + DISTS
*达芬奇*

## 实现效果

我们采用的算法在A榜排名第1，得分0.5678，提交request_id 2022062923314973979051；B榜排名第3，得分0.5570，提交request_id 2022071220561848633908，

<img src="https://github.com/22TonyFStark/
SemanticGeneration_da_Vinci/raw/main/selects/3481474497_8409ea7057_b.jpg" width="300">
<img src="https://github.com/22TonyFStark/
SemanticGeneration_da_Vinci/raw/main/selects/5951555151_13dbd1c42e_b.jpg" width="300">
<img src="https://github.com/22TonyFStark/
SemanticGeneration_da_Vinci/raw/main/selects/7646157822_18126d0db6_b.jpg" width="300">


## 简介
本项目包含了第二届计图挑战赛计图 - 草图生成风景比赛的代码实现。本项目的特点是：我们采用Jittor框架复现了UNITE，并针对判别器复现了SESAME，最后复现了DISTS loss，超越了原始UNITE方法的效果。

有任何问题可以联系qingzhongfei21@mails.ucas.ac.cn

## 安装 
本项目可在 4 张 3090 上运行，训练时间约为 3~4天。

#### 运行环境（修改）
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖，jittor框架请参考官网的安装知道
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型模型下载地址为 https:abc.def.gh，下载后放入目录 `<root>/weights/` 下。

## 数据预处理
在本次比赛中，我们没有采用额外的数据预处理和数据增强操作，在项目代码中已经有所体现

预训练模型采用了jittor框架自带的vgg16和vgg19模型，无需额外下载，代码运行会自动下载。

## 训练
风景图像生成网络包括两步：  
1.训练UNITE模型150epoch：
`bash train.sh`   
2.选择第1步中在测试集上最好的结果，读入此参数，训练UNITE+Sesame模型60epoch：  
(1).将train.sh里的--niter设为30，--niter_decay也设为30；  
(2).将train.sh里的--which_epoch设为第1步中最好的epoch（参考：85~120）；   
(3).将models/networks/discriminator.py重命名为discriminator_original.py，然后将discriminator_sesame.py重命名为discriminator.py，并执行：
`bash train.sh`

## 推理
1. 请确保模型参数位于正确的位置：

`checkpoints/UNITE_eqlrsn_nce_dists_sesame_x256`: 30_net_Corr.pkl / 30_net_G.pkl

`weights`: EsrganG.pkl

2. 请修改test.sh中的：

   trainimg_root:全部训练图片路径

   input_path:测试集语义图片路径

   output_path:输出保存路径，默认是results

3. 运行命令：
   bash:  
   `bash test.sh`
   python:  

   ```python
   CUDA_VISIBLE_DEVICES=1 python test.py \
   --name UNITE_eqlrsn_nce_dists_sesame_x256 \
   --dataset_mode scene \
   --correspondence 'ot' \
   --nThreads 0 \
   --use_attention \
   --maskmix \
   --warp_mask_losstype direct \
   --PONO \
   --PONO_C \
   --eqlr_sn  \
   --adaptor_nonlocal \
   --batchSize 4 \
   --aspect_ratio 1 \
   --im_height 256 \
    --im_width 256 \
   --gpu_ids 0 \
   --which_epoch 30 \
    --trainimg_root '/home/qingzhongfei/A_scene/SPADE/datasets/train/train/all_img' \
    --input_path 'data/B/' \
    --output_path 'results' \
    --no_pairing_check 
   ```

## 致谢
本项目主要参考如下：

* [欢迎查阅计图文档 — Jittor 1.3.4.10 文档 (tsinghua.edu.cn)](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
* [fnzhan/UNITE: Unbalanced Feature Transport for Exemplar-based Image Translation [CVPR 2021\] and Marginal Contrastive Correspondence for Guided Image Generation [CVPR 2022] (github.com)](https://github.com/fnzhan/UNITE)
* [piq/perceptual.py at 0c23870615cb16eb3d909663eee30d7a538d7b76 · photosynthesis-team/piq (github.com)](https://github.com/photosynthesis-team/piq/blob/0c23870615cb16eb3d909663eee30d7a538d7b76/piq/perceptual.py)
* [Jittor/JGAN: JGAN model zoo supports 27 kinds of mainstream GAN models with high speed for jittor. (github.com)](https://github.com/Jittor/JGAN)
* [OpenSESAME/discriminator.py at master · vglsd/OpenSESAME (github.com)](https://github.com/vglsd/OpenSESAME/blob/master/models/networks/discriminator.py)


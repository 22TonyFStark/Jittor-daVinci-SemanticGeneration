python train.py \
  --name debug \
  --dataset_mode scene \
  --dataroot 'data/debug_dataset/' \
  --correspondence 'ot' \
  --display_freq 2000 \
  --niter 75 \
  --niter_decay 75 \
  --maskmix \
  --aspect_ratio 1.3333 \
  --use_attention \
  --warp_mask_losstype direct \
  --weight_mask 100.0 \
  --PONO \
  --PONO_C \
  --adaptor_nonlocal \
  --ctx_w 0.5 \
  --gpu_ids 1 \
  --batchSize 1  \
  --label_nc 29 \
  --mcl  \
  --skip_corr  \
  --dists_w 0 \
  --checkpoints_dir checkpoints

# /home/qingzhongfei/A_scene/SPADE/datasets/train/train/
# data/debug_dataset/
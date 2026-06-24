
CUDA_VISIBLE_DEVICES=1 python examples/train_ema.py \
-d market1501 --eps 0.5 --use-hard --height 256 \
--self-norm --loss_weight 0.04 --epochs 40 --epochs1 20 --k 3 --stage1 --stage2 \
-pp /data/vit_small_cfs_lup.pth \
 --logs-dir ../log/market1501

CUDA_VISIBLE_DEVICES=1 python examples/train_ema.py \
-d market1501 --eps 0.5 --use-hard -a vit_base --height 256 \
--self-norm --loss_weight 0.04 --epochs 40 --epochs1 30 --k 3 --conv-stem --stage1 --stage2 \
-pp /data/vit_base_ics_cfs_lup.pth \
 --logs-dir ../log/market1501-vit-base

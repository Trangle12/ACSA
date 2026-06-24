
CUDA_VISIBLE_DEVICES=2 python examples/train_ema.py \
-d msmt17 --eps 0.7 -a vit_small --height 256 \
--self-norm --loss_weight 0.04 --loss_weight1 0.5 --epochs 10 --epochs1 40 --k 4 --stage1 --stage2 \
-pp /data/vit_small_cfs_lup.pth --eval-step 10 \
--logs-dir ../log-response/msmt17


CUDA_VISIBLE_DEVICES=2 python examples/train_ema.py \
-d msmt17 --eps 0.7 -a vit_base --height 256 \
--self-norm --loss_weight 0.04 --loss_weight1 0.5 --epochs 40 --epochs1 60 --k 5  --conv-stem --stage1 --stage2 \
-pp /data/vit_base_ics_cfs_lup.pth \
--logs-dir ../log/msmt17-vit-base








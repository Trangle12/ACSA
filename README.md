<p align="center">
  <h1 align="center"></h1>

# Mask-guided Asymmetric Contrastive and Semantic Alignment for Unsupervised Person Re-Identification

## Installation

Install `conda` before installing any requirements.

```bash

conda create -n acsa python=3.9
conda activate acsa
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install tqdm
conda install scikit-learn

pip install yacs
pip install timm
pip install scikit-image
pip install ftfy
pip install easydict
pip install regex
pip install faiss-gpu

```

## Datasets

Make a new folder named `data` under the root directory. Download the datasets and unzip them into `data` folder.
* [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
* [MSMT17](https://arxiv.org/abs/1711.08565)

## Training

For example, training the full model on Market1501 with GPU 0 and saving the log file and checkpoints to `logs/market-pclclip`:

```
CUDA_VISIBLE_DEVICES=0 python examples/train_ema.py -d market1501 --eps 0.5 --use-hard --height 256 --self-norm --loss_weight 0.04 --epochs 40 --epochs1 20 --k 3 --stage1 --stage2 -pp /data/vit_small_cfs_lup.pth --logs-dir ../log/market1501

CUDA_VISIBLE_DEVICES=0 python examples/train_ema.py -d market1501 --eps 0.5 --use-hard -a vit_base --height 256 --self-norm --loss_weight 0.04 --epochs 40 --epochs1 30 --k 3 --conv-stem --stage1 --stage2 -pp /data/vit_base_ics_cfs_lup.pth --logs-dir ../log/market1501-vit-base

CUDA_VISIBLE_DEVICES=1 python examples/train_ema.py -d msmt17 --eps 0.7 -a vit_small --height 256 --self-norm --loss_weight 0.04 --loss_weight1 0.5 --epochs 10 --epochs1 40 --k 4 --stage1 --stage2 -pp /data/vit_small_cfs_lup.pth --eval-step 10 --logs-dir ../log-response/msmt17

CUDA_VISIBLE_DEVICES=1 python examples/train_ema.py -d msmt17 --eps 0.7 -a vit_base --height 256 --self-norm --loss_weight 0.04 --loss_weight1 0.5 --epochs 40 --epochs1 60 --k 5  --conv-stem --stage1 --stage2 -pp /data/vit_base_ics_cfs_lup.pth --logs-dir ../log/msmt17-vit-base
```

## Results

The results are on Market1501 (M) and MSMT17 (MS). 
| Methods | Backbone | M | Link | MS | Link |
| --- | -- | -- | -- | -- | - |
| Ours | ViT-S/16 | 90.7 (96.2) | [model](https://drive.google.com/file/d/1qjQcmASFnbKPYK_TLFbHvzWgrPPUSCSf/view?usp=sharing) | 58.2 (81.9) | [model](https://drive.google.com/file/d/1qjQcmASFnbKPYK_TLFbHvzWgrPPUSCSf/view?usp=sharing) |
| Ours | ViT-B/16 | 92.2 (96.5) | [model](https://drive.google.com/file/d/1aOUQ-NmdKZx6KZ4u7rDqNO3P6p_Ror_9/view?usp=sharing) | 64.5 (85.4) | [model](https://drive.google.com/file/d/1aOUQ-NmdKZx6KZ4u7rDqNO3P6p_Ror_9/view?usp=sharing) |


# Acknowledgements

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[TransReID-SSL](https://github.com/damo-cv/TransReID-SSL), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid), [MAE](https://github.com/facebookresearch/mae), [UntransReID](https://github.com/mangye16/ReID-Survey/tree/master/Transformer-ReID-Survey/UnTransReID_USL_ReID), [TMGF](https://github.com/RikoLi/WACV23-workshop-TMGF).


# Not All Unlabeled Data are Equal:<br/> Learning to Weight Data in Semi-supervised Learning

## Overview
This code is for paper:
[Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning](https://arxiv.org/pdf/2007.01293v1.pdf). Zhongzheng Ren*, Raymond A. Yeh*, Alexander G. Schwing. arXiv:2007.01293. (*equal contribtion)

## License

Copyright (C) 2020 The Paper Authors. All Rights Reserved.

The code is released for internal use. Please don't distribute as the paper is still under review.
 
## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details. <br>
**Environement***: this code is tested using python-3.7, anaconda3-5.0.1, cuda-10.0, cudnn-v7.6, tensorflow-1.14/1.15

### Install dependencies

```bash
conda create -n semi-sup python=3.7
conda activate semi-sup
pip install -r requirements.txt
```
make sure `tf.test.is_gpu_available() == True` after installation so that GPUs will be used.

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord
    done
done
```

## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, train a model with 32 filters on cifar10 shuffled with `seed=1`, 250 labeled samples and 1000 validation sample:
```bash
# single-gpu
CUDA_VISIBLE_DEVICES=0 python main.py --filters=32 --dataset=cifar10.1@250-1000 --train_dir ./experiments

# multi-gpu: just pass more GPUs and the model automatically scales to them, here we assign GPUs 0-1 to the program:
CUDA_VISIBLE_DEVICES=0,1 python main.py --filters=32 --dataset=cifar10.1@250-1000 --train_dir ./experiments
```

**Naming rule**: `${dataset}.${seed}@${size}-${valid}`<br>
Available labelled sizes are 250, 1000, 4000.<br>
For validation, available sizes are 1000, 5000.<br>
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).

### Image classification
The hyper-parameters used in the paper:
```bash
for SEED in 1 2 3 4 5; do
    for SIZE in 250 1000 4000; do
    CUDA_VISIBLE_DEVICES=0,1 python main.py --filters=32 
        --dataset=cifar10.${SEED}@${SIZE}-1000 \
        --train_dir ./experiments --alpha 0.01 --inner_steps 512
    done
done
```

### Flags

```bash
python main.py --help
# The following option might be too slow to be really practical.
# python main.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml main.py
```

### Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

### Checkpoint accuracy

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.1@250-1000, extracting accuracy:
./scripts/extract_accuracy.py ./experiments/cifar10.d.d.d.1@250-1000/CTAugment_depth2_th0.80_decay0.990/FixMatch_alpha0.01_archresnet_batch64_confidence0.95_filters32_inf_warm0_inner_steps100_lr0.03_nclass10_repeat4_scales3_size_unlabeled49000_uratio7_wd0.0005_wu1.0
# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Use you own data

## Citing this work
If you use this code for your research, please cite our paper.
```bibtex
@inproceedings{ren-yeh-ssl2020,
  title = {Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning},
  author = {Zhongzheng Ren$^\ast$ and Raymond A. Yeh$^\ast$ and Alexander G. Schwing},
  booktitle = {arXiv:2007.01293},
  year = {2020},
  note = {$^\ast$ equal contribution},
}
```

## Acknowledgement

The code is built based on:
[FixMatch (commit: 08d9b83)](https://github.com/google-research/fixmatch)

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685). Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.


## Contact
Github issues and PR are preferred. Feel free to contact Jason Ren (zr5 AT illinois.edu) for any questions!

# Not All Unlabeled Data are Equal:<br/> Learning to Weight Data in Semi-supervised Learning

## Overview
This is the code for FixMatch variant in NeurIPS 2020 submission 2536. This folder contains the following:

*   Code for image classifications on CIFAR-10 and SVHN (Sec.3.2).

## Reproducibility
All of the code in this repository used `NVIDIA V100 Tensor Core GPUs`. We used multi-GPU training and two GPUs are used for each experiments.

## Installation & Dataset

Please check the `README_FixMatch.md` for detailed instructions; all the settings the same for a fair comparison. We have experimented using `cifar10` and `svhn_noextra` with `250, 1000, and 4000` labeled samples in our paper.

## Image classification (ours)
Please check the bash scripts under `./runs`.

## Acknowledgement

The code is built based on:
[FixMatch (commit: 08d9b83)](https://github.com/google-research/fixmatch)

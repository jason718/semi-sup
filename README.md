# Not All Unlabeled Data are Equal:<br/> Learning to Weight Data in Semi-supervised Learning

## Overview
This repo is for arXiv paper:
```bibtex
@inproceedings{ren-yeh-ssl2020,
  title = {Not All Unlabeled Data are Equal: Learning to Weight Data in Semi-supervised Learning},
  author = {Zhongzheng Ren$^\ast$ and Raymond A. Yeh$^\ast$ and Alexander G. Schwing},
  booktitle = {arXiv:2007.01293},
  year = {2020},
  note = {$^\ast$ equal contribution},
}
```

## License

Copyright (C) 2020 The Paper Authors. All Rights Reserved.

The code is released for internal use. Please don't distribute as the paper is still under review.
 
## How to use?

### Installation & Dataset

Please check the `README_FixMatch.md` for detailed instructions; all the settings the same for a fair comparison. We have experimented using `cifar10` and `svhn_noextra` with `250, 1000, and 4000` labeled samples in our paper.

### Image classification
Please check the bash scripts under `./runs`.

### Use you own data

## Acknowledgement

The code is built based on:
[FixMatch (commit: 08d9b83)](https://github.com/google-research/fixmatch)

## Contact
Github issues and PR are preferred. Feel free to contact Jason Ren (zr5 AT illinois.edu) for any questions!

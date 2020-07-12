# This script is kept the same as in original code:
# https://github.com/google-research/fixmatch.
# Due to the hardware difference, our reproduced 
# results are comparable but not identical.
# Therefore, we report both in our submission.

# CIFAR-10
for SEED in 1 2 3 4 5; do
    for SIZE in 250 1000 4000; do
    CUDA_VISIBLE_DEVICES=0,1 python fixmatch.py \
        --dataset=cifar10.${SEED}@${SIZE}-1 \
        --filters=32 --train_dir OUTPUT_DIR
    done
done

# SVHN
for SEED in 1 2 3 4 5; do
    for SIZE in 250 1000 4000; do
    CUDA_VISIBLE_DEVICES=0,1 python fixmatch.py \
        --dataset=svhn_noextra.${SEED}@${SIZE}-1 \
        --filters=32 --train_dir OUTPUT_DIR
    done
done
# CIFAR-10
for SEED in 1 2 3 4 5; do
    for SIZE in 250 1000 4000; do
    CUDA_VISIBLE_DEVICES=0,1 python fixmatch_inf.py \
        --filters=32 --dataset=cifar10.${SEED}@${SIZE}-1000 \
        --train_dir OUTPUT_DIR --alpha 0.01 --inner_steps 512
    done
done

# SVHN
for SEED in 1 2 3 4 5; do
    for SIZE in 250 1000 4000; do
    CUDA_VISIBLE_DEVICES=0,1 python fixmatch_inf.py \
        --filters=32 --dataset=svhn_noextra.${SEED}@${SIZE}-1000 \
        --train_dir OUTPUT_DIR --alpha 0.01 --inner_steps 512
    done
done
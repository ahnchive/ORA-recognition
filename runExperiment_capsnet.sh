#!/bin/bash

# declare -a expnames=("test" "capsnet_clean_run1" "capsnet_clean_run2" "capsnet_clean_run3" "capsnet_clean_run4" "capsnet_clean_run5")
declare -a expnames=("test" "capsnet_shift_run1" "capsnet_shift_run2" "capsnet_shift_run3" "capsnet_shift_run4" "capsnet_shift_run5")

for rseed in 1 2 3 4 5; do
#     python train_capsnet.py --cuda 1 --task mnist --seed $rseed --expname ${expnames[$rseed]}
    python train_capsnet.py --cuda 0 --task mnist_shift --seed $rseed --expname ${expnames[$rseed]}
done


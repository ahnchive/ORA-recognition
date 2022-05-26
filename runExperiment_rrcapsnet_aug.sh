#!/bin/bash

declare -a expnames=("test" "clean_aug_run1" "clean_aug_run2" "clean_aug_run3" "clean_aug_run4" "clean_aug_run5")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 0 --task mnist --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]}
done

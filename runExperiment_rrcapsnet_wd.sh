#!/bin/bash

declare -a expnames=("test" "clean_wd2_run1" "clean_wd2_run2" "clean_wd2_run3" "clean_wd2_run4" "clean_wd2_run5")

# for rseed in 1 2 3 4 5; do
for rseed in 3 4 5; do
#     python train_rrcapsnet.py --cuda 1 --task mnist --clr False --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]} 
    python train_rrcapsnet.py --cuda 0 --task mnist --clr False --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]} --max_lr 0.0005
done

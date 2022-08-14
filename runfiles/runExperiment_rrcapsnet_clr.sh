#!/bin/bash

declare -a expnames=("test" "clean_clr_run1" "clean_clr_run2" "clean_clr_run3" "clean_clr_run4" "clean_clr_run5")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 0 --task mnist --clr True --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]}
done

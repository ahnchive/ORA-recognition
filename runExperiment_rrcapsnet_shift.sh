#!/bin/bash

declare -a expnames=("test" "shift_run1" "shift_run2" "shift_run3" "shift_run4" "shift_run5")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 1 --task mnist_shift --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]}
done

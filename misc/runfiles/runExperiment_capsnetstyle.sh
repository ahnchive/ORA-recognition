#!/bin/bash

declare -a expnames=("test" "run1_capsnetstyle" "run2_capsnetstyle" "run3_capsnetstyle" "run4_capsnetstyle" "run5_capsnetstyle")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 0 --task mnist_recon --seed $rseed --time_step 1 --routings 1 --epoch 50 --expname ${expnames[$rseed]}
done


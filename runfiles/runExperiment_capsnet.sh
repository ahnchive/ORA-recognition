#!/bin/bash

declare -a expnames=("test" "capsnet1" "capsnet2" "capsnet3" "capsnet4" "capsnet5")
for rseed in 1 2 3 4 5; do
    python train_capsnet.py --cuda 0 --task mnist --seed $rseed --expname ${expnames[$rseed]}
done


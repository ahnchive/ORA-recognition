#!/bin/bash


declare -a expnames=("test" "recon_run1" "recon_run2" "recon_run3" "recon_run4" "recon_run5")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 1 --task mnist_recon --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]}
done
#!/bin/bash


# declare -a expnames=("test" "cnn_clean_run1" "cnn_clean_run2" "cnn_clean_run3" "cnn_clean_run4" "cnn_clean_run5")
declare -a expnames=("test" "cnn_shift_run1" "cnn_shift_run2" "cnn_shift_run3" "cnn_shift_run4" "cnn_shift_run5")

for rseed in 1 2 3 4 5; do
#      python train_cnn.py --cuda 0 --task mnist --seed $rseed --expname ${expnames[$rseed]}
     python train_cnn.py --cuda 0 --task mnist_shift --seed $rseed --expname ${expnames[$rseed]}

done



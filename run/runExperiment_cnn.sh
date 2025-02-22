#!/bin/bash


# declare -a expnames=("test" "cnn_clean_run1" "cnn_clean_run2" "cnn_clean_run3" "cnn_clean_run4" "cnn_clean_run5")
# declare -a expnames=("test" "cnn_shift_run1" "cnn_shift_run2" "cnn_shift_run3" "cnn_shift_run4" "cnn_shift_run5")
# declare -a expnames=("test" "cnn_aug_run1" "cnn_aug_run2" "cnn_aug_run3" "cnn_aug_run4" "cnn_aug_run5")
# declare -a expnames=("test" "cnn_res_run1" "cnn_res_run2" "cnn_res_run3" "cnn_res_run4" "cnn_res_run5")

# declare -a expnames=("test" "cnn_res4_run1" "cnn_res4_run2" "cnn_res4_run3" "cnn_res4_run4" "cnn_res4_run5")
declare -a expnames=("test" "cnn_res4_2fc_run1" "cnn_res4_2fc_run2" "cnn_res4_2fc_run3" "cnn_res4_2fc_run4" "cnn_res4_2fc_run5")

for rseed in 1 2 3 4 5; do
#      python train_cnn.py --cuda 0 --task mnist --seed $rseed --expname ${expnames[$rseed]}
#      python train_cnn.py --cuda 0 --task mnist_shift --seed $rseed --expname ${expnames[$rseed]}
     python train_cnn.py --cuda 1 --task mnist --seed $rseed --expname ${expnames[$rseed]}

done



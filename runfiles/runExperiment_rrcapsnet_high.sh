#!/bin/bash


# declare -a expnames=("test" "recon_run1" "recon_run2" "recon_run3" "recon_run4" "recon_run5")
# declare -a expnames=("test" "recon_edge_run1" "recon_edge_run2" "recon_edge_run3" "recon_edge_run4" "recon_edge_run5")
# declare -a expnames=("test" "train_edge_run1" "train_edge_run2" "train_edge_run3" "train_edge_run4" "train_edge_run5")

# declare -a expnames=("test" "blur_resnet_run1" "blur_resnet_run2" "blur_resnet_run3" "blur_resnet_run4" "blur_resnet_run5")

# declare -a expnames=("test" "blur_res_run1" "blur_res_run2" "blur_res_run3" "blur_res_run4" "blur_res_run5")
# declare -a expnames=("test" "blur_res4_run1" "blur_res4_run2" "blur_res4_run3" "blur_res4_run4" "blur_res4_run5")
# declare -a expnames=("test" "hsf_res4_run1" "hsf_res4_run2" "hsf_res4_run3" "hsf_res4_run4" "hsf_res4_run5")
declare -a expnames=("test" "hsf_conv_run1" "hsf_conv_run2" "hsf_conv_run3" "hsf_conv_run4" "hsf_conv_run5")

for rseed in 1 2 3 4 5; do
# for rseed in 5; do
    python train_rrcapsnet.py --cuda 1 --task mnist_recon_high --seed $rseed --time_step 1 --routings 1 --expname ${expnames[$rseed]}
done
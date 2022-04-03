#!/bin/bash
# for rrcapsnet, change timestep, inputreconmatch version accordingly
# for rseed in 0 1 2 3 4 5 6 7 8 9; do
#     for timestep in 1 3; do
#         for userecon in True False; do
#         python train_rrcapsnet.py --task mnist --seed $rseed --timestep 1 --usereconloss $userecon --inputreconmatch nomatch 
#         python train_rrcapsnet.py --task mnist_recon --time_steps 3 --routings 3 --use_recon_decoder $userecon --seed $rseed
#             python train_rrcapsnet.py --task mnist_recon --time_steps $timestep --use_recon_decoder $userecon --seed $rseed
#         done
#     done
# done
declare -a expnames=("test" "run1" "run2" "run3" "run4" "run5")

for rseed in 1 2 3 4 5; do
    python train_rrcapsnet.py --cuda 0 --task mnist_recon --seed $rseed --time_step 1 --routings 1 --epoch 50 --expname ${expnames[$rseed]}
    
done



# for original capsnet
# for rseed in 0 1 2 3 4; do
#     python capsulenet.py --task mnist --seed $rseed
# done
#     for matchtype in nomatch subtract add; do


# for num in {100..104..1}; do

## declare an array variable for expnames
# declare -a expnames=("3step_recurrent_4" "3step_recurrent_5" "3step_recurrent_6")

# get length of an array
# ntrial=${#expnames[@]}

# use for loop to read all values and indexes
# for (( i=0; i<${ntrial}; i++ )); do
#      python runcapdraw.py --cuda 3 --expname ${expnames[$i]} --task "multimnist" --timestep 3 --useattn "False" --epoch 100
# done

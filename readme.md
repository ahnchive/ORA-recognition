The current repo is currently under updates (last update: Feb 7 2023)

# Reconstruction-guided attention on MNIST-C
Many visual phenomena suggest that humans use top-down generative or reconstructive processes to create visual percepts (e.g., imagery, object completion, pareidolia), but little is known about the role reconstruction plays in robust object recognition. We built an iterative encoder-decoder network that generates an object reconstruction and used it as top-down attentional feedback to route the most relevant information. This repository contains codes to train and test our model on MNIST-C dataset. Our model showed strong generalization performance on this dataset, on average outperforming all other models including feedforward CNNs and adversarially trained networks. More details can be found in this [paper](https://openreview.net/forum?id=tmvg0VIHTDr)


To cite this work:
```
@inproceedings{
ahn2022reconstructionguided,
title={Reconstruction-guided attention improves the robustness and shape processing of neural networks},
author={Seoyoung Ahn and Hossein Adeli and Greg Zelinsky},
booktitle={SVRHM 2022 Workshop @ NeurIPS },
year={2022},
url={https://openreview.net/forum?id=tmvg0VIHTDr}
}
```

# set environment
Following packages are required, full list in `requirements.txt`. Tested with python =3.8 and pytorch =1.7
- python == 3.8 
- pytorch == 1.7 
- torchvision==0.8.2
- ipykernel ==6.29
- pandas == 2.0.3
- matplotlib = 3.7.5
- tensorboardX==2.6
- tqdm
- prettytable


# Command for training models
`bash run/runExperiment_our.sh`


# For model evaluation and visualizations
- Please copy and place the ipython notebook from `notebook` in the current folder. 
- You can load pretrained models from `models`

# Resources
Origianl capsnet implementation is from https://github.com/XifengGuo/CapsNet-Pytorch

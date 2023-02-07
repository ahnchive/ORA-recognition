The current repo is currently under updates (last update: Feb 7 2023)

# Reconstruction-guided attention on MNIST-C
Many visual phenomena suggest that humans use top-down generative or reconstructive processes to create visual percepts (e.g., imagery, object completion, pareidolia), but little is known about the role reconstruction plays in robust object recognition. We built an iterative encoder-decoder network that generates an object reconstruction and used it as top-down attentional feedback to route the most relevant information. This repository contains codes to train and test our model on MNIST-C dataset. Our model showed strong generalization performance on this dataset, on average outperforming all other models including feedforward CNNs and adversarially trained networks. More details can be found in this [paper](https://openreview.net/forum?id=tmvg0VIHTDr)


To cite this work:
```
@inproceedings{ahn2022reconstruction,
  title={Reconstruction-guided attention improves the robustness and shape processing of neural networks},
  author={Ahn, Seoyoung and Adeli, Hossein and Zelinsky, Greg},
  booktitle={SVRHM 2022 Workshop@ NeurIPS}
}
```

# Command for training models
`bash run/runExperiment_our.sh`


# For model evaluation and visualizations
Please copy and place the ipython notebook from `notebook` in the current folder. 

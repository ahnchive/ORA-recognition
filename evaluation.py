import torch

# ------------------
# Accuracy
# ------------------

def topkacc(y_pred: torch.Tensor, y_true:  torch.Tensor, topk=1, only_acc=True):
    """
    how many of topk predictions are accurate --> 1 (all topk are in labels), 0.5 (half topk are in lables), 0 (none)
    e.g., one of top2 is in gt label = 0.5    

    Input: 
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        topk_values, topk_indices = y_pred.topk(topk, sorted=True)
        accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)
        accs = (accs/topk)

    if only_acc:
        return accs
    else:
        return accs, topk_values

def exactmatch(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    if y_pred and y_true matches exactly --> 1, not --> 0
    
    Input: torch tensor 
        - both y_pred and y_true should be in the same format
        e.g., if y_true is multi-hot, then y_pred should be made in multi-hot as well
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        accs = (y_pred == y_true).all(dim=1).float()
    
    return accs

def partialmatch(y_pred: torch.Tensor, y_true:  torch.Tensor, n_targets=2):
    """
    when n_targets=2, if one of the two predictions are accurate --> 0.5, none--> 0
    
    Input: 
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        topk_indices = y_pred.topk(n_targets, sorted=True)[1] 
        accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)/n_targets

    return accs


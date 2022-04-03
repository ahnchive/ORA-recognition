# --------------------
# RRcapsnet
# --------------------
# code Refs:
# modified from https://github.com/XifengGuo/CapsNet-Pytorch
# modified from https://github.com/kamenbliznashki/generative_models/blob/master/draw.py

import os
import sys
import time
from tqdm import tqdm
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils import *
from loaddata import *


# --------------------
# functions & modules
# --------------------

    
def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    
    return scale * inputs


class nnCapsulate(nn.Module):
    '''
    Note that input.size(0) is usually the batch size.
    Given any input with input.size(0) # of batches,
    features will be capsulated [batch, num_caps, dim_caps] + squash
    '''
    def __init__(self, num_caps, dim_caps):
        super().__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
    
    def forward(self, input):
        batch_size = input.size(0)
        out = input.permute(0, 2, 3, 1).contiguous().view(batch_size, self.num_caps, self.dim_caps) #contiguous
        return squash(out) 

# class nnSquash(nn.Module):
#     '''
#     pytorch module version of squash
#     '''
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, input):
#         return squash(input) 


class Encoder(nn.Module):
    """ Encoding process is two steps:
        1) convolution: feature extraction (either using pretrained modle or train-from-scratch)
        2) encoder: transform/convert extracted features to capsulizable size (either recurrent or just one-step linear way)
        
        Input:
        - x = image # x.size = (x.shape[0], -1, self.read_size, self.read_size) # should on the device
        Outputs:
        - primary caps  [batch, num_caps, dim_caps] 
    """
    def __init__(self, img_channels, dim_caps):
        super().__init__()
        self.dim_caps = dim_caps
        
        # define encoder type

        self.enc_feature_shape = (32*8, 6, 6)
        enc_feature_size = self.enc_feature_shape[0]*self.enc_feature_shape[1]*self.enc_feature_shape[2]
        self.num_caps = int(enc_feature_size/self.dim_caps) #(6*6)*256/8

        self.enc = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=img_channels, out_channels=256, kernel_size=9, stride=1, padding=0)),
            ('conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=9, stride=2, padding=0)),
            ('capsulate', nnCapsulate(dim_caps=self.dim_caps, num_caps=self.num_caps))
        ]))
                            

        print(f"...resulting primary caps #: {self.num_caps}, dim: {self.dim_caps}")

    def forward(self, x):
        batch_size = x.size(0)
        pricaps = self.enc(x)
        
        return pricaps

class Decoder(nn.Module):
    """ Decoding process is two steps:
        1) decoder: transform/convert capsule features to original feature dimension (either recurrent or one-step linear)
        2) deconvolution: image reconstruction 
    
        Input:
        - object capsules # size  
        
        Output:
        - x_recon: image reconstruction
        - if decoder is recurrent, hidden_state and cell_state (for lstm)
    """
    def __init__(self, outcaps_size, output_shape, enc_feature_shape):
        super().__init__()

        self.outcaps_size = outcaps_size # num_objcapsule * dim_objcapsule         
        
        # input shape
        self.C, self.H, self.W = output_shape
        
        # encoded features shape
        self.nfeature, self.fH, self.fW = enc_feature_shape 

        
        # define decoder

        self.dec = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('delinear1', nn.Linear(self.outcaps_size, 512)),
            ('derelu1', nn.ReLU()),
            ('decoder-dropout1', nn.Dropout(0.5)),
            ('delinear2', nn.Linear(512, 1024)),
            ('derelu2', nn.ReLU()),
            ('decoder-dropout2', nn.Dropout(0.5)),
            ('delinear3', nn.Linear(1024, self.C*self.H*self.W)),
#                 ('derelu3', nn.ReLU())
            ('sigmoid', nn.Sigmoid())
        ]))            
            
          
    
    def forward(self, outcaps):     
        batch_size = outcaps.size(0)
        outcaps = outcaps.contiguous().view(batch_size, -1) # flatten all object capsules into a single vector
        
        # decoder
        x_recon = self.dec(outcaps)
        
        return x_recon.view(batch_size, self.C, self.H, self.W)     





class OrigCapsuleRouting(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings):
        super().__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        assert self.routings > 0, 'The \'routings\' should be > 0.'
        self.weight = nn.Parameter(0.1 * torch.randn(self.out_num_caps, self.in_num_caps, self.out_dim_caps, self.in_dim_caps))


    def forward(self, incaps):
        device = incaps.device
        batch_size = incaps.shape[0]
        
        # initialize coefficients 
        self.b = torch.zeros(batch_size, self.out_num_caps, self.in_num_caps, device=device) # part-whole matching
        self.c = torch.ones(batch_size, self.out_num_caps, self.in_num_caps, device=device) # normalized b
        self.c = F.softmax(self.c, dim=1)

        # incaps.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1, in_num_caps, in_dim_caps,  1]
        # weight.size   =[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => outcaps_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        outcaps_hat = torch.squeeze(torch.matmul(self.weight, incaps[:, None, :, :, None]), dim=-1)
        
        # In forward pass, `outcaps_hat_detached` = `outcaps_hat`;
        # In backward, no gradient can flow from `outcaps_hat_detached` back to `outcaps_hat`.
        outcaps_hat_detached = outcaps_hat.detach()

        # keep all the coupling coefficients for all routing steps 
        coups = []; betas = []
        for i in range(self.routings):
            # record coups and betas
            coups.append(self.c.detach())
            betas.append(self.b.detach())
            
            # Use `outcaps_hat_detached` to update `b`. No gradients flow on this path.       
            if i < self.routings-1: 
                outcaps = squash(torch.sum(self.c[:, :, :, None] * outcaps_hat_detached, dim=-2, keepdim=True))
                           
                # update beta and coupling coefficients
                # outcaps.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # outcaps_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => resulting b.size          =[batch, out_num_caps, in_num_caps]
                self.b = self.b + torch.sum(outcaps * outcaps_hat_detached, dim=-1) ## todo; whether accumulate?                
                self.c = F.softmax(self.b, dim=1)
#                 self.c = scale_coef(self.b, dim=1)

               
            # At last iteration, use `outcaps_hat` to compute `outcaps` in order to backpropagate gradient
            elif i == self.routings-1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # outcaps_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outcaps.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outcaps = squash(torch.sum(self.c[:, :, :, None] * outcaps_hat, dim=-2, keepdim=True))
                break

             
        outputs = {}
        outputs['coups'] = coups
        outputs['betas'] = betas
        outcaps_squeezed = torch.squeeze(outcaps, dim=-2) # [batch, out_num_caps, out_dim_caps]
        return outcaps_squeezed, outputs

    

#########################
# Model build and Training
#########################

class CapsNet(nn.Module):
    """
    todo: describe this model
    This model 
        1) encodes an image through recurrent read operations that forms a spatial attention on objects
        2) effectively binds features and classify objects through capsule representation and its dynamic routing
        3) decodes/reconstructs an image from class capsules through recurrent write operations 
    """
        
    def __init__(self, args):
        super().__init__()
        
        print(f'\n=========== model instantiated like below: =============')
        # dataset info
        print(f'TASK: {args.task} (# targets: {args.num_targets}, # classes: {args.num_classes}, # background: {args.backg_objcaps})')
        self.task = args.task
        self.C, self.H, self.W = args.image_dims
        self.image_dims = args.image_dims
        self.num_targets = args.num_targets # the number of objects in the image
        self.num_classes = args.num_classes # number of categories 
        

        
        # encoder (given image --> pricaps)
        self.encoder = Encoder(img_channels= self.C,
                               dim_caps = args.dim_pricaps,
                              )
        

        # capsule network (pricaps --> objcaps)
        print(f'ROUTINGS # {args.routings}')
        self.dim_pricaps = args.dim_pricaps
        self.num_pricaps = self.encoder.num_caps
        self.num_objcaps = args.num_classes + args.backg_objcaps
        

        self.capsule_routing = OrigCapsuleRouting(in_num_caps= self.num_pricaps, 
                                            in_dim_caps= self.dim_pricaps,
                                            out_num_caps= self.num_objcaps,
                                            out_dim_caps= args.dim_objcaps, 
                                            routings= args.routings)

        # decoder (objcaps--> reconstruction)
        self.use_decoder = args.use_decoder # whether use decoder for reconstruction 

        print(f'DECODER: {self.use_decoder}')
        self.clip_recon = args.clip_recon
        if self.use_decoder: 
            self.decoder = Decoder(outcaps_size= self.num_objcaps*args.dim_objcaps,
                                   output_shape= (self.C, self.H, self.W), 
                                   enc_feature_shape = self.encoder.enc_feature_shape
                                  ) 
    
            self.reconstruct_only_one_capsule = args.reconstruct_only_one_capsule # or reconstruct based on all object capsules
            print(f'...recon only one object capsule: {self.reconstruct_only_one_capsule}')
        
        
        print(f'========================================================\n')
      
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        device = x.device

        #############
        # encoding feature
        #############
        pricaps= self.encoder(x) # h_enc is used only when encoder use rnn projection

        objcaps, _ =  self.capsule_routing(pricaps) # objectcaps [n_batch, n_objects(class+bkg), dim_objectcaps] 

        objcaps_len = objcaps.norm(dim=-1)  #[n_batch, n_objects]


        ############
        # decoding reconstruction
        ############
        if self.use_decoder:

            ####################
            # whether use one most likely capsule for reconstruction
            ###################
            if self.reconstruct_only_one_capsule:

                # get background recon
                if self.num_objcaps > self.num_classes:
                    y_bkg = torch.zeros(objcaps_len.size(), device= objcaps_len.device)
                    y_bkg[:, self.num_classes:]= 1
                    objcaps_bkg = (objcaps * y_bkg[:, :, None])
                    x_recon_bkg, _ = self.decoder(objcaps_bkg, h_dec)
                else:
                    x_recon_bkg = torch.zeros(x.shape, device=device)

                # get most likely obj recon
                objcaps_len_narrow = objcaps_len.narrow(dim=1,start=0, length=self.num_classes)
                idx_max_obj = objcaps_len_narrow.max(dim=1)[1]
                # idx_max_obj = obj_rscore.max(dim=1)[1]
                y_onehot = torch.zeros(objcaps_len.size(), device= objcaps_len.device).scatter_(1, idx_max_obj.view(-1,1), 1.)
                objcaps_onehot = (objcaps * y_onehot[:, :, None])
                x_recon_obj = self.decoder(objcaps_onehot) # h_dec is only used when decoder use rnn projection

                # get final recon
                x_recon = x_recon_obj + x_recon_bkg

            else: # or recon from all capsules combined
                x_recon = self.decoder(objcaps) # h_dec is only used when decoder use rnn projection                    


            if self.clip_recon:
                x_recon = torch.clip(x_recon, min=0.0, max=1.0)


                
        return objcaps_len, x_recon


# ------------------
# Loss functions
# ------------------
def margin_loss(y_pred, y_true):
    '''
    margin loss is used for classification loss
    '''
    # narrow down to valid objects (excluding bkg capsule output)
    num_classes = y_true.size(dim=1)
    y_pred = y_pred.narrow(dim=1,start=0, length=num_classes) 
    
    # calculate losses
    m_neg = 0.1 # margin loss allowed for negative case (for absent digits)
    lam_abs = 0.5 # down-weighting loss for absent digits (prevent the initial learning from shrinking the lengths of the class capsules    
    L_present =  y_true* torch.clamp((y_true-m_neg) - y_pred, min=0.) ** 2   
#     L_present =  torch.clamp(y_true, min=0., max=1.) * torch.clamp((y_true-m_neg) - y_pred, min=0.) ** 2 # clamped version  
    L_absent = lam_abs * torch.clamp(1 - y_true, min=0.) * torch.clamp(y_pred-m_neg, min=0.) ** 2
    L_margin = (L_present+L_absent).sum(dim=1).mean()
    
    return L_margin



def mse_loss(x_recon, x, clip=False):
    '''
    mse loss is used for reconstruction loss
    '''
    if clip:  # for clipping cumulative recon canvas, not necessary for individual recon  
        x_recon = torch.clip(x_recon,0,1)
    return nn.MSELoss()(x_recon, x)



def loss_fn(objcaps_len, y_true, x_recon, x, args, gtx=None, use_recon_loss=True):
 
    if torch.is_tensor(gtx): # if separate grountruth x (intact version) is given
        x = gtx # replace x with gtx    
    
    if args.use_decoder:
        class_loss = margin_loss(objcaps_len, y_true)
        recon_loss = mse_loss(x_recon, x, clip=False)
        total_loss = class_loss + args.lam_recon*recon_loss

    else:
        recon_loss = torch.Tensor([-99])
        total_loss = class_loss
    return total_loss, class_loss, recon_loss


# ------------------
# Accuracy
# ------------------

def topkacc(y_pred: torch.Tensor, y_true:  torch.Tensor, topk=1):
    """
    if one of the top2 predictions are accurate --> 1, none--> 0
    
    Input: 
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        topk_indices = y_pred.topk(topk, sorted=True)[1] 
        accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)

    return accs

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

def acc_fn(objcaps_len, y_true, acc_name):
    '''
    1) topk accuracy: format should be top@k,
    2) 
    '''
    num_classes = y_true.size(dim=1)
    # get final prediction
    y_pred = objcaps_len
    y_pred = y_pred.narrow(dim=1,start=0, length=num_classes) # in case a background cap was added    
#     y_pred = torch.sum(objcaps_len_step, dim=1)
#     y_pred = y_pred.narrow(dim=1,start=0, length=num_classes) # in case a background cap was added    
    
    if 'top' in acc_name:
        topk = int(acc_name.split('@')[1])
        accs = topkacc(y_pred, y_true, topk=topk)
    else:
        raise NotImplementedError('given acc functions are not implemented yet')
        
    return accs
        
# --------------------
# Train and Test
# --------------------

def train_epoch(model, train_dataloader, optimizer, epoch, writer, args):
    """
    for each batch:
        - forward pass  
        - compute loss
        - param update
    """    
    losses = AverageMeter('Loss', ':.4e')
    
    model.train() 
    with tqdm(total=len(train_dataloader), desc='epoch {} of {}'.format(epoch, args.n_epochs)) as pbar:
#     time.sleep(0.1)        
        
        # load batch from dataloader 
        for i, data in enumerate(train_dataloader):
            global_step = (epoch-1) * len(train_dataloader) + i + 1 #global batch number
            
            # load dataset on device
#             x = x.view(x.shape[0], -1).to(args.device)
            if len(data)==2:
                x, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = None
            elif len(data)==3:
                x, gtx, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = gtx.to(args.device)

            # forward pass
            objcaps_len_step, x_recon_step = model(x)
            
            # compute loss for this batch and append it to training loss
            loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
                                 gtx=gtx, use_recon_loss = args.use_decoder) 
            # separate class/recon
#             if epoch<50:
#                 loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
#                                      gtx=gtx, use_recon_loss = False)
#             elif epoch>=50:
#                 loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
#                                      gtx=gtx, use_recon_loss = args.use_decoder) 

            losses.update(loss.item(), x.size(0))



            # minibatch update; zero out previous gradients and backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # record grad norm and clip to prevent exploding gradients
            if args.record_gradnorm:
                grad_norm = 0
                for name, p in model.named_parameters():
                    grad_norm += p.grad.norm().item() if p.grad is not None else 0
                writer.add_scalar('grad_norm', grad_norm, global_step)                
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # update param
            optimizer.step()

            # end of each batch, update tqdm tracking
            pbar.set_postfix(batch_loss='{:.3f}'.format(loss.item()))
            pbar.update()
    
    return losses.avg


def evaluate(model, x, y, args, acc_name, gtx=None):
    """
    Run model prediction on testing dataset and compute loss/acc 
    """
    
    # evaluate
    model.eval()
    
    # load testing dataset on device
#     x = x.view(x.shape[0], -1).float().to(args.device)
    x= x.to(args.device)                
    y = y.to(args.device)
    
    if torch.is_tensor(gtx):
        gtx = gtx.to(args.device)

    with torch.no_grad():
        
        # run model with testing data and get predictions
        objcaps_len_step, x_recon_step = model(x)
        
        # compute batch loss and accuracy
        loss, loss_class, loss_recon = loss_fn(objcaps_len_step, y, x_recon_step, x, args, gtx=gtx)            
        acc = acc_fn(objcaps_len_step, y, acc_name).mean()

    return (loss, loss_class, loss_recon), acc, objcaps_len_step, x_recon_step


def test(model, dataloader, args, acc_name):
    """
    for each batch:
        - evaluate loss & acc ('evaluate')
    log average loss & acc  
    """   
    losses = AverageMeter('Loss', ':.4e')
    losses_class = AverageMeter('Loss_class', ':.4e')
    losses_recon = AverageMeter('Loss_recon', ':.4e')
    accs_topk = AverageMeter(acc_name, ':6.2f')
    
    # load batch data
    for data in dataloader:
        if len(data)==2:
            x, y = data
            gtx = None
        elif len(data)==3:
            x, gtx, y = data
            
        # evaluate
        batch_losses, batch_acc, objcaps_len_step, x_recon_step, \
        =  evaluate(model, x, y, args, acc_name, gtx)
            
        # aggregate loss and acc
        losses.update(batch_losses[0].item(), x.size(0))
        losses_class.update(batch_losses[1].item(), x.size(0))
        losses_recon.update(batch_losses[2].item(), x.size(0))
        accs_topk.update(batch_acc.item(), x.size(0))

    return losses.avg, losses_class.avg, losses_recon.avg, accs_topk.avg


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, writer, args, acc_name):
    """
    for each epoch:
        - train the model, update param, and log the training loss ('train_epoch')
        - save checkpoint
        - compute and log average val loss/acc
        - save best model
    """
    start_epoch = 1

    if args.restore_file:
        print(f'Restoring parameters from {args.restore_file}')
        start_epoch = load_checkpoint(args.restore_file, [model], [optimizer], map_location=args.device.type)
        args.n_epochs += start_epoch
        print(f'Resuming training from epoch {start_epoch}')
    
    path_best = None
    
    for epoch in range(start_epoch, args.n_epochs+1):
        
        # train epoch and lod to tensorboard writer

        train_loss = train_epoch(model, train_dataloader, optimizer, epoch, writer, args)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        scheduler.step()
        
        # save checkpoint 
        if args.save_checkpoint:
            save_checkpoint({'epoch': epoch,
                             'model_state_dicts': [model.state_dict()],
                             'optimizer_state_dicts': [optimizer.state_dict()]},
                            checkpoint=args.log_dir,
                            quiet=True)

        # compute validation loss and acc
        if (epoch) % args.validate_after_howmany_epochs == 0:
            val_loss, val_loss_class, val_loss_recon, val_acc = test(model, val_dataloader, args, acc_name)
            
            # logging validation info to tensorboard writer
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Loss_Class', val_loss_class, epoch)
            writer.add_scalar('Val/Locc_Recon', val_loss_recon, epoch)
            writer.add_scalar('Val/Acc', val_acc, epoch)
                        
            if args.verbose:
                print(f"==> Epoch {epoch:d}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_loss_class={val_loss_class:.5f}, val_loss_recon={val_loss_recon:.5f}, val_acc={val_acc:.4f}")
               
            # update best validation acc and save best model to output dir
            if (val_acc > args.best_val_acc):
                args.best_val_acc = val_acc
                # remove previous best
                if path_best:
                    try:
                        os.remove(path_best)
                    except:
                        print("Error while deleting file ", path_best)

                # save current best
                path_best = args.log_dir +f'/best_model_epoch{epoch:d}_acc{val_acc:.4f}.pt'
                torch.save(model.state_dict(), path_best)  
                print(f"the model with best val_acc ({val_acc:.4f}) was saved to disk")
                
        # archive models        
        if (epoch%10==0):  
            torch.save(model.state_dict(), args.log_dir +f'/archive_model_epoch{epoch:d}_acc{val_acc:.4f}.pt')  #output_dir
            print(f"model archived at epoch ({epoch})")            


        # abort the training if...
        if (epoch%100 == 0) and (epoch < args.n_epochs):
            if hasattr(args, 'abort_if_valacc_below'):
                if (args.best_val_acc < args.abort_if_valacc_below) or math.isnan(val_acc):
                    
                    # save aborted model as final
                    torch.save(model.state_dict(), args.log_dir +f'/final_model_epoch{epoch:d}_acc{val_acc:.4f}.pt')
                    status = f'===== EXPERIMENT ABORTED: val_acc is {val_acc} at epoch {epoch} (Criterion is {args.abort_if_valacc_below}) ===='
                    writer.add_text('Status', status, epoch)
                    print(status)
                    sys.exit()
                else:
                    status = '==== EXPERIMENT CONTINUE ===='
                    writer.add_text('Status', status, epoch)
                    print(status)
        

        # save final model
        if (epoch == args.n_epochs):
            torch.save(model.state_dict(), args.log_dir +f'/final_model_epoch{epoch:d}_acc{val_acc:.4f}.pt')


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


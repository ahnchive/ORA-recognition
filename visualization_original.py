import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from rrcapsnet_original import get_every_obj_rscore, scale_coef

def plot_imgarray(imgarray, row_title=None, col_title=None, row_text =None, fontsize=20, **imshow_kwargs):
    # row title, col title both should be vectors 
    
    if not isinstance(imgarray, np.ndarray):
        imgarray = np.array(imgarray)
    if not len(np.array(imgarray).shape)==5:
        raise ValueError('input images should be list or array with shape of (nrows, ncols, H, W, C)')

    num_rows, num_cols, h, w, c = imgarray.shape
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            img = imgarray[row_idx,col_idx]
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
    if col_title is not None:
        for ax, col in zip(axs[0], col_title):
            ax.set_title(col, fontsize=fontsize, color='black')
    
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set_ylabel(row_title[row_idx], fontsize=fontsize, color='black')

    if row_text is not None:
        for row_idx in range(num_rows):
            axs[row_idx, num_cols-1].text(w+1, h, row_text[row_idx], fontsize=fontsize, color='black')
            
    plt.tight_layout()
    plt.show()
    
def plot_coef(coef, objlen_before, objrscore, objlen, num_classes):
    import seaborn as sns
    
    nrow = len(coef) # routing steps    
    nobj = coef[-1].shape[0] # number of objects (classes + bkg)
    nprimary = coef[-1].shape[1] # number of primary caps
    fig, axs = plt.subplots(nrow, 4, figsize=(20,4*nrow), gridspec_kw={'width_ratios': [8, 1, 1, 1]})

    if nobj > num_classes:
        objlabels = [i for i in range(num_classes)]+ ['BG' for i in range(nobj-num_classes)]
    else:
        objlabels = [i for i in range(num_classes)]
                
    for n in range(nrow):
        # coefficient array heatmap
        sns.heatmap(coef[n].round(2),
                    cmap="Greys",  # Choose a squential colormap
                    annot=True, # Place values on the heatmap
                    annot_kws={'fontsize':7},  # Reduce size of label to fit
                    fmt='.2f',          # Interpret labels as strings
                    square=True,     # Force square cells
                    vmax=1.0,         # Ensure same 
                    vmin=0.5**2,          # color scale
                    linewidth=0.01,  # Add gridlines
                    linecolor="black",# Adjust gridline color
                    ax=axs[n,0],        # Arrange in subplot
                    cbar=False,
                   )


#         axs[n,0].set_title(f'routing {n+1}', color='blue')
        axs[n,0].set_ylabel(f'object caps @ routing {n+1}', fontsize=15)
        axs[n,0].set_yticklabels(objlabels) 
        axs[n,0].set_xlabel('primary caps')
        axs[n,0].set_xticklabels([i for i in range(1,nprimary+1)])
        

        
        ## plot routing variables
        ylabels = [i for i in range(len(objlen[n]))]
        
        # objlen before rerror
        axs[n,1].barh(ylabels,  objlen_before[n].round(2),  height=1.0, facecolor='grey', edgecolor='black')
        axs[n,1].set_yticklabels(objlabels) 
        axs[n,1].invert_yaxis()
        axs[n,1].set_ylabel('object')
        axs[n,1].set_xlabel('Raw Class Likelihood') # capsule length before rscore applied, i.e., simple dynamic routing
        axs[n,1].set_yticks(ylabels)
        axs[n,1].set_xticks([0, 0.5, 1.0])
        axs[n,1].set_xlim(0, 1.0)
        axs[n,1].spines['right'].set_visible(False)
        axs[n,1].spines['top'].set_visible(False)
        axs[n,1].margins(x=0, y=0)
        
        # rerror
        axs[n,2].barh(ylabels,  objrscore[n][:,0].round(2),  height=1.0, facecolor='grey', edgecolor='black')
        axs[n,2].set_yticklabels(objlabels) 
        axs[n,2].invert_yaxis()
        axs[n,2].set_ylabel('object')
        axs[n,2].set_xlabel('Reconstruction Score')
        axs[n,2].set_yticks(ylabels)
#         axs[n,2].set_xticks([0, 0.5, 1.0])
#         axs[n,2].set_xlim(0, 1.0)
        axs[n,2].spines['right'].set_visible(False)
        axs[n,2].spines['top'].set_visible(False)
        axs[n,2].margins(x=0, y=0)

        #final class likelihood
        axs[n,3].barh(ylabels, objlen[n].round(2), height=1.0, facecolor='grey', edgecolor='black')
        axs[n,3].set_yticklabels(objlabels) 
        axs[n,3].invert_yaxis()
        axs[n,3].set_ylabel('object')
        axs[n,3].set_xlabel('Adjusted Class Likelihood') # capsule length after rscore applied
        axs[n,3].set_yticks(ylabels)
        axs[n,3].set_xticks([0, 0.5, 1.0])
        axs[n,3].set_xlim(0, 1.0)
        axs[n,3].spines['right'].set_visible(False)
        axs[n,3].spines['top'].set_visible(False)
        axs[n,3].margins(x=0, y=0)



    plt.tick_params(axis='both', labelsize=12)
    fig.suptitle('coupling coeffs', x=0.1)
    plt.tight_layout()
    plt.show()
        # plt.savefig('final.png', dpi=120)
    
def plot_capsules(imgarray, max_obj_step, col_title, row_title, col_text=None, fontsize=15, **imshow_kwargs):
    from matplotlib.patches import Rectangle
    # row title, col title both should be vectors 
    # plot_objrecon()
    if not isinstance(imgarray, np.ndarray):
        imgarray = np.array(imgarray)
    if not len(np.array(imgarray).shape)==5:
        raise ValueError('input images should be list or array with shape of (nrows, ncols, H, W, C)')

    num_rows, num_cols, h, w, _ = imgarray.shape
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            img = imgarray[row_idx,col_idx]
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    for row_idx in range(num_rows):
        axs[row_idx, max_obj_step[row_idx]+2].add_patch(Rectangle((0, 0), w-1, h-1, edgecolor = 'red', fill=False, lw=5))

    if col_title is not None:
        for ax, col in zip(axs[0], col_title):
            ax.set_title(col, fontsize=fontsize, color='black')

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set_ylabel(row_title[row_idx], fontsize=fontsize, color='black')

    if col_text is not None:
        for row_idx in range(len(col_text)):
            for col_idx in range(num_cols):
                axs[row_idx,col_idx].text(int(w/2), h+12, col_text[row_idx][col_idx], fontsize=12, color='gray', ha='center')

    plt.tight_layout()
    plt.show()



def visualize_detail(model, x, y, outputs, x_recon_step, objcaps_len_step, args, start=0, n_image=100, plot_trials_when='all', plot_routings = False, pred_to_compare=None, num_steps_to_finish=None, only_plot_object=None):
    DEVICE =x.device 
    num_objcaps = args.num_classes + args.backg_objcaps
    
    ####################################
    # get recon and rscore for each obj
    ####################################
    obj_rscore_step, obj_recon_step = [],[]
    for t in range(1, args.time_steps+1):
        # get objcaps and length (activation) at each step
        x_input = outputs['x_input'][:,t-1]  # torch.Size([1000, 1, 28, 28])
        objcaps = outputs['objcaps'][:,t-1].squeeze() # torch.Size([1000, 10, 32])
        
#         if args.use_STN:
#             affine_params = outputs['affine_params'][:,t-1]
#             # get recon without affine transformed
#             _, obj_recon = get_every_obj_rscore(x_input, objcaps,  model.decoder, scale=True, save_recon=True, affine_inv=None, affine_params=None) #  torch.Size([1000, 10]), torch.Size([1000, 10, 1, 28, 28])
#             # get rscore with affine transformed
#             obj_rscore = get_every_obj_rscore(x_input, objcaps,  model.decoder, scale=True, save_recon=False, affine_inv= model.stn.affine_inv, affine_params=affine_params) #  torch.Size([1000, 10]), torch.Size([1000, 10, 1, 28, 28])
        obj_rscore, obj_recon = get_every_obj_rscore(x_input, objcaps,  model.decoder, save_recon=True) #  torch.Size([1000, 10]), torch.Size([1000, 10, 1, 28, 28])
        obj_rscore = scale_coef(obj_rscore, dim=1)
        
        obj_rscore_step.append(obj_rscore)
        obj_recon_step.append(obj_recon)
    
    obj_rscore_step = torch.stack(obj_rscore_step, dim=2)  ## torch.Size([1000, 10, 3])
    obj_recon_step = torch.stack(obj_recon_step, dim=2)  ## torch.Size([1000, 10, 3, 1, 28, 28])
    
    ####################################
    # creating image array for plot
    ####################################
    objrecon_plotdata = [x.repeat(1,args.time_steps,1,1).unsqueeze(2), x_recon_step] + [obj_recon_step[:,i] for i in range(num_objcaps)] # each element torch.Size([1000, 3, 1, 28, 28])
    objrecon_plotdata = torch.stack(objrecon_plotdata, dim=1) # torch.Size([1000, 12, 3, 1, 28, 28])

    ##########################################
    # for labels 
    ########################################
    # softmax objcaps length
    objcaps_len_step_narrow = objcaps_len_step.narrow(dim=2,start=0, length=args.num_classes)
#     objcaps_prob_step = F.softmax(objcaps_len_step, dim=-1) #torch.Size([1000, 3, 10])
    # obj identity with max activation
    max_act_obj = objcaps_len_step_narrow.max(dim=-1)[1] #torch.Size([1000, 3])
    
    ##############################
    # plot each trial in the batch
    ###############################
    count = 0
    
    for idx in range(start, start+n_image):
        if num_steps_to_finish:
            timesteps = num_steps_to_finish[idx]
        else:
            timesteps = args.time_steps
            
        # get gt and pred label info 
        gt = y[idx].argmax(dim=0).cpu().item()
        baseline = pred_to_compare[idx].item()
        max_obj_step = list(max_act_obj[idx].cpu().detach().numpy()) # (3,)
        correct_step =[]
        text_step = []
        for t in range(timesteps):
#             ps = objcaps_len_step[idx,i:i+1].squeeze(1)
#             ps = ps.argmax(dim=1).cpu().item()
            ps = max_obj_step[t]
            correct = gt==ps
            correct_step.append(correct)
            text = f'**CORRECT**: \ngt:{gt}, \nbaseline pred: {baseline}, \nour pred:{ps}' if correct else f'INCORRECT: \ngt:{gt}, \nbaseline pred: {baseline}, \nour pred:{ps}'
            text_step.append(text)
        
        # plot conditional on...
        if only_plot_object:
            if gt==only_plot_object:
                pass
            else:
                continue

        if type(plot_trials_when) == list:
            if idx in plot_trials_when:
                pass
            else:
                continue
        elif plot_trials_when == 'all':
            pass
        elif plot_trials_when == 'first correct last incorrect':
            if correct_step[0] and not correct_step[-1]:
                pass
            else:
                continue
        elif plot_trials_when == 'first incorrect last correct':
            if correct_step[-1] and not correct_step[0]:
                pass
            else:
                continue
        elif plot_trials_when == 'first incorrect last incorrect':
            if not correct_step[-1] and not correct_step[0]:
                pass
            else:
                continue
        elif plot_trials_when == 'correct':
            if correct_step[-1]:
                pass
            else:
                continue
        elif plot_trials_when == 'incorrect':
            if not correct_step[-1]: 
                pass
            else:
                continue
        
        count += 1
        
        ############################
        # plot input vs recon at each stepwise
        #############################
        print(f'\n\n================ TRIAL {idx} ===================')
        # create image array (n_step, (orig, mask, input, recon))
        orig = x[idx].cpu().data.numpy()
        x_origs = np.array([orig for i in range(timesteps)])
        x_masks = outputs['x_mask'][idx][:timesteps].cpu().data.numpy() #timesteps x channel x height x width array
        x_inputs = outputs['x_input'][idx][:timesteps].cpu().data.numpy()
        x_recons = x_recon_step[idx][:timesteps].cpu().data.numpy()
        imgarray = np.stack((x_origs, x_masks, x_inputs,x_recons), axis=1) #timesteps x 4 x channel x height x width array
        

        plt.rcParams["figure.figsize"] = (10,2*timesteps)
        row_title = [f'step{i+1}' for i in range(timesteps)]
        col_title = ['original', 'attn mask', 'masked input', 'recon']    
        img = plot_imgarray(np.transpose(imgarray, [0, 1, 3, 4, 2]), row_title=row_title, col_title=col_title, row_text=text_step)
        
        #######################################
        # plot stepwise capsule representation
        #######################################
        # create image array (x, recon, obj123...)
        imgarray = objrecon_plotdata.cpu().detach().numpy()[idx][:,:timesteps] # (12, 3, 1, 28, 28)

        col_text = []
        
        for t in range(timesteps):
            objlen = list(objcaps_len_step[idx].cpu().detach().numpy()[t].round(2))
        #     objprob = list(objcaps_prob_step[bid].cpu().detach().numpy()[t].round(2))
            
            objrerror = list(obj_rscore_step[idx].cpu().detach().numpy()[:,t].round(2))
            col_text.append(['GT: '+str(gt) + '\n\n'] + ['PRED: '+str(max_obj_step[t]) + '\n\n'] + ['C: '+ str(objlen[i]) + '\nR: '+ str(objrerror[i]) + '\n' for i in range(len(objlen))] )
        #     col_text.append(['', ''] + ['len: '+ str(objlen[i]) + '\nprob: '+ str(objprob[i]) + '\n' for i in range(len(objlen))] )

        plt.rcParams["figure.figsize"] = (20,2.2*timesteps)
        row_title = ['step'+str(i+1) for i in range(timesteps)]
        
        if args.backg_objcaps:
            col_title = ['orig', 'recon'] + ['obj'+str(i) for i in range(args.num_classes)] + ['bkg'+str(i) for i in range(args.backg_objcaps)]
        else:
            col_title = ['orig', 'recon'] + ['obj'+str(i) for i in range(args.num_classes)]
        
        plot_capsules(np.transpose(imgarray, [1, 0, 3, 4, 2]), max_obj_step, row_title=row_title, col_title=col_title, col_text=col_text)
        
        ##################################
        #plot coupling coefficient
        ##################################
        if plot_routings:
            if args.routings>1:
                for t in range(timesteps):
                    print('coupling coeff at time step: ', t+1)
                    coup_everyrouting = outputs['coups'][idx][t].cpu().numpy()
                    beta_everyrouting = outputs['betas'][idx][t].cpu().numpy()
                    recon_coup_everyrouting = outputs['recon_coups'][idx][t].cpu().numpy()

                    objrscore_everyrouting = outputs['rscores'][idx][t].cpu().numpy()
                    objlen_everyrouting = outputs['outcaps_len'][idx][t].cpu().numpy()
                    objlen_before_everyrouting = outputs['outcaps_len_before'][idx][t].cpu().numpy()

                    print("self.rc===")
                    plot_coef(recon_coup_everyrouting[:,:,:40], objlen_before_everyrouting, objrscore_everyrouting[:,:,:40], objlen_everyrouting, num_classes = args.num_classes)

    print(f'\n\n FINISED. There are {count} images plotted')
        

# just list of imgs --> nrow*ncol
# def plot(imgarray, num_rows = 1, row_title=None, col_title=None, fontsize=15, **imshow_kwargs):

#     num_cols = int(len(imgs)/num_rows) + (len(imgs)%num_rows > 0)
#     fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

#     for row_idx in range(num_rows):
#         for col_idx, img in enumerate(imgs[row_idx*num_cols:(row_idx+1)*num_cols]):
#             ax = axs[row_idx, col_idx]
#             ax.imshow(np.asarray(img), cmap='gray_r', **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set_ylabel(row_title[row_idx], fontsize=fontsize, color='black')

#     if col_title is not None:
#         for row_idx in range(len(col_title)):
#             for col_idx in range(num_cols):
#                 axs[row_idx,col_idx].set_xlabel(col_title[row_idx][col_idx], fontsize=fontsize, color='black')
            
#     plt.tight_layout()
#     plt.show()

    
    
# def combine_images(generated_images, nrow=2):
#     import math
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     height = int(math.ceil(float(num)/width))
#     shape = generated_images.shape[1:3]
#     image = np.zeros((height*shape[0], width*shape[1]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index/width)
#         j = index % width
#         image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
#             img[:, :, 0]
#     return image

def plot(imgarray, row_title=None, col_title=None, fontsize=15, **imshow_kwargs):
    # row title is vector, col title can be array
    
    if not isinstance(imgarray, np.ndarray):
        imgarray = np.array(imgarray)
    if not len(np.array(imgarray).shape)==5:
        raise ValueError('input images should be list or array with shape of (nrows, ncols, H, W, C)')

    num_rows, num_cols, _, _, _ = imgarray.shape
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            img = imgarray[row_idx,col_idx]
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set_ylabel(row_title[row_idx], fontsize=fontsize, color='black')

    if col_title is not None:
        for row_idx in range(len(col_title)):
            for col_idx in range(num_cols):
                axs[row_idx,col_idx].set_xlabel(col_title[row_idx][col_idx], fontsize=fontsize, color='black')
            
    plt.tight_layout()
    plt.show()



    
def visualize_batch(x, y, output_step, objcaps_len_step, include_sum=False, start=0, n_image=10):
    # get label info to get col titles
    gt = y[start:start+n_image].argmax(dim=1).cpu().data
    gt_label = [f'gt:{gt[i]}' for i in range(len(gt))]
    time_steps = len(output_step[0])

    pred_step_label = []
    for i in range(time_steps):
        ps = objcaps_len_step[start:start+n_image,i:i+1].squeeze(1)
        ps = ps.argmax(dim=1).cpu().data
        ps_label = [f'*pred:{ps[i]}' if gt[i]==ps[i] else f'pred:{ps[i]}' for i in range(len(ps))]
        pred_step_label.append(ps_label)

    # create image array
    orig = x[start:start+n_image].numpy()
    recon_step = []

    if time_steps >1:
        for i in range(time_steps):
            recon = output_step[start:start+n_image,i:i+1].squeeze(1).cpu().data.numpy()
            recon_step.append(recon)

    if include_sum:
        recon_summed = torch.sum(output_step,dim=1)[start:start+n_image].cpu().data.numpy()            
        listcompared = [orig] +  recon_step + [recon_summed]

        row_title = ['orignal'] + ['step'+str(i+1) for i in range(time_steps)] + ['sum_over_steps'] 
        pred_sum = torch.sum(objcaps_len_step,dim=1)[start:start+n_image].argmax(dim=1).cpu().data
        pred_sum_label = [f'*pred:{pred_sum[i]}' if gt[i]==pred_sum[i] else f'pred:{pred_sum[i]}' for i in range(len(pred_sum))]

        col_title = [gt_label] + pred_step_label + [pred_sum_label]
    else:
        listcompared = [orig] + recon_step
        row_title = ['orignal'] + ['step'+str(i+1) for i in range(time_steps)]
        col_title = [gt_label] + pred_step_label
    #     col_title = [f'gt:{gt[i]} pred:{pred[i]}' for i in range(len(gt))]
    #     col_title = ['*'+col if gt[i]==pred[i] else col for i, col in enumerate(col_title)]
    imgarray = np.array(listcompared)
    plt.rcParams["figure.figsize"] = (20,3*time_steps)
    img = plot(np.transpose(imgarray, [0, 1, 3, 4, 2]), row_title=row_title, col_title=col_title)

        

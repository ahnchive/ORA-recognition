
#     if 'top' in acc_type:
#         # get final prediction
#         y_pred = objcaps_len_step_narrow[:,-1]
#         topk = int(acc_type.split('@')[1])
#         accs = topkacc(y_pred, y_true, topk=topk)
        
#     elif acc_type =='hypothesis':
#         # hypothesis testing 2 consecutive --> correct
#         if single_step:
#             # get final prediction
#             y_pred = objcaps_len_step_narrow[:,-1]
#             accs = topkacc(y_pred, y_true, topk=1)
#         else:

#             def get_first_zero_index(x, axis=1):
#                 cond = (x == 0)
#                 return ((cond.cumsum(axis) == 1) & cond).max(axis, keepdim=True)[1]

#             pstep = objcaps_len_step_narrow.max(dim=-1)[1]
#             pnow = pstep[:,1:]
#             pbefore = pstep[:,:-1]

#             pdiff = (pnow-pbefore)
#             null_column = -99*torch.ones(pdiff.size(0),1).to(pdiff.device)
#             pdiff = torch.cat([null_column, pdiff], dim=1) # first step as null
#             pdiff[:,-1]=0 # final step as zero
#             nstep = get_first_zero_index(pdiff)
#             y_pred= torch.gather(pstep, 1, nstep).flatten()
#             accs = torch.eq(y_pred, y_true.max(dim=1)[1]).float()
        
#     elif acc_type == 'entropy'
#         from torch.distributions import Categorical

#         def get_first_true_index(boolarray, axis=1, when_no_true='final_index'):
#             # boolarray = Batch x Stepsize
#             first_true_index = ((boolarray.cumsum(axis) == 1) & boolarray).max(axis, keepdim=True)[1] # when no true, set as 0

#             if when_no_true == 'final_index': # when there is no true, use final index
#                 final_index = boolarray.shape[1]-1
#                 no_true_condition = (~boolarray).all(dim=1).reshape(-1,1)
#                 first_true_index = first_true_index + final_index * no_true_condition
#                 return  first_true_index, no_true_condition
#             else:
#                 return first_true_index

#         if use_cumulative:
#             score = objcaps_len_step_narrow.cumsum(dim=1)
#             pred = score.max(dim=-1)[1]
#         else:
#             score = objcaps_len_step_narrow # Batch x Stepsize x Category
#             pred = score.max(dim=-1)[1] # Batch x Stepsize


#         # compute entropy from softmax output with Temp scale
#         T=0.2
#         softmax = F.softmax(score/T, dim=-1) # torch.Size([1000, 4, 10])
#         entropy = Categorical(probs = softmax).entropy() # torch.Size([1000, 4])

#         # entropy thresholding
#         stop = entropy<threshold
#         boolarray = (stop == True)

#         # get first index that reached threshold
#         first_true_index, no_stop_condition = get_first_true_index(boolarray, axis=1, when_no_true='final_index')

#         final_pred = torch.gather(pred, dim=1, index= first_true_index).flatten()
#         accs = torch.eq(final_pred.cpu(), y_hot.max(dim=1)[1])
#         nstep = (first_true_index.flatten()+1).cpu().numpy()

#         return nstep, final_pred, acc, no_stop_condition

#     nstep, pred_model, acc_model_check, no_stop_condition = get_nstep(objcaps_len_step_narrow, y_hot, threshold=1.0, use_cumulative = False)
    
#     else:
#         raise NotImplementedError('given acc functions are not implemented yet')

import os
from tqdm.notebook import tqdm
import pprint
import time

import torch

from utils import *
from loaddata import *

from ourmodel import * 
# from rrcapsnet_writeglimpse import * 

import warnings
# warnings.filterwarnings('ignore')


#######################################
# parse experiments args <<-- added for experiments
#######################################
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

## required arguments
parser.add_argument('--cuda', '-c', help="cuda index", type= int, required=True)
parser.add_argument('--task', '-t', help="task type", type= str, required=True)
parser.add_argument('--param_file', help="model hyper parameters", type= str, required=True)

## optional arguments
parser.add_argument('--lr', help="lr", type= float, default=0.001) #0.0005
parser.add_argument('--epoch',  help="epoch n", type= int, default=1000)
parser.add_argument('--batch', help="training batch size", type= int, default=128)
parser.add_argument('--seed', help="seed for random", type= int, default=1)
parser.add_argument('--expname', help="name of the experiment (will be added to results path)", type= str, default=None)
parser.add_argument('--clr', help="use onecyle learning or not", type= str2bool, default=False)
# parser.add_argument('--max_lr', help="maxlr", type= float, default=0.001)

## others
parser.add_argument('--print', action='store_true', help="if true, just print model info, false continue training", default=False)
parser.add_argument('--lrfind', action='store_true', help="if true, run lr range test", default=False)
## required training parameter
parser.add_argument('--time_steps', help="number of recurrent steps", type=int, required=True)
parser.add_argument('--routings', help="num of dynamic routings", type= int, required=True)

## optional training parameter -- default is all True
parser.add_argument('--use_decoder', help="whether to reconstruct images", type= str2bool, default=True)
parser.add_argument('--num_bgcapsule', help="# of background capsule?", type= int, default=0)


expargs = parser.parse_args()

############################################
# set hyperparams using param file 
############################################
# load the default param setting for the current task
params_filename = expargs.param_file
args = parse_params(params_filename)
args.task = expargs.task

# if you have a checkpoint to restore, specify restore file (in the orginal param file or here)
# args.restore_file =  'results/multimnist_cluttered/Aug28_4014__step7_1/state_checkpoint.pt'
if args.restore_file: 
    # if you want to pick up from save-point, reload param files
    path_savepoint = os.path.dirname(args.restore_file)
    params_filename = path_savepoint + '/params.txt'    
    assert os.path.isfile(params_filename), "No param flie exists"
    print(f"param file reloaded from {path_savepoint}")
#     args = parse_params(params_filename)     
    args = parse_params_wremove(params_filename, removelist = ['device']) # because device does not go through literal_eval
    
    # reassign path_savepoint to restorefile
    args.restore_file = path_savepoint + '/state_checkpoint.pt'

# setup output directory where log folder should be created 
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

###########################
# set experimental arguments before instantiate model
############################


# training setup
args.n_epochs= expargs.epoch
args.batch_size= expargs.batch
args.lr = expargs.lr
# args.max_lr = expargs.max_lr
args.clr= expargs.clr

# seed for reproducibility
args.rand_seed = expargs.seed
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.rand_seed)

# device
args.device = torch.device('cuda:{}'.format(expargs.cuda) if torch.cuda.is_available() and expargs.cuda is not None else 'cpu')


# training parameter
args.time_steps = expargs.time_steps
args.routings = expargs.routings

if expargs.use_decoder == False:
    args.use_decoder = False
    
    # set decoder related param to None
    args.reconstruct_only_one_capsule = None
    args.lam_recon = None
    args.use_reconstruction_mask = None
else:
    args.use_decoder=True
    
args.backg_objcaps = expargs.num_bgcapsule

#set up experiment name --> will be used for results path
if expargs.expname:
    COMMENT = expargs.expname
else:
    c1 = 'recon' if args.use_decoder else 'norecon'
    c2 = 'step' + str(args.time_steps)
#     c3 = 'nocomparison' if not args.input_recon_comparison else str(args.input_recon_comparison)
#     c3 = str(args.recon_type)
    c3 = 'routing' + str(args.routings)
    c4 = 'seed'+str(args.rand_seed)
    COMMENT = '_'.join([c1, c2, c3, c4])     
    
    
############################################
# instantiate model 
############################################
print(f'\nSTARTS TRAINING on task:{args.task}, device:{args.device}, expname: {COMMENT}')

# set up model and optimizer
model = RRCapsNet(args).to(args.device) 


###########################
# main (either print model info, find lr range, or train model)
##########################
if expargs.print:
    print('\n==> print argument file')
    pprint.pprint(args.__dict__, sort_dicts=False)
    
    print('\n==> print model architecture')
    print(model)
    
    print('\n==> print model params')
    count_parameters(model)

elif expargs.lrfind:
    # load training param for lr range test
    args.n_epochs = 10
    args.batch_size = 128
    args.lr = 0.000001
    
    # load dataloader 
    train_dataloader, _ = fetch_dataloader(args.task, args.data_dir, args.device, args.batch_size, train=True)

    print('\n==> start lr range test begins')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
    lr_range_test(model, train_dataloader, optimizer, scheduler, args, acc_name='top@1')

else:
    # load dataloader 
    train_dataloader, val_dataloader = fetch_dataloader(args.task, args.data_dir, args.device, args.batch_size, train=True)

    # set writer for tensorboard
    writer, current_log_path = set_writer(log_path = args.output_dir if args.restore_file is None else args.restore_file,
                        comment = COMMENT, 
                        restore = args.restore_file is not None) 
    args.log_dir = current_log_path



    # set optimizer and scheduler 
    print(f'decaying lr scheduler is used, starts from: {args.lr}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96) #0.96

#     if args.clr:
#         print('cycle lr scheduler is used')
#         print(f'lr cycles btw:{args.lr} , {args.max_lr}')

#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_dataloader), epochs=args.n_epochs)    
#     else:
#         print('decaying lr scheduler is used')
#         print(f'lr starts from: {args.max_lr}')
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr) 
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96) #0.98
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    
    # save used param info to writer and logging directory for later retrieval
    writer.add_text('Params', pprint.pformat(args.__dict__))
    with open(os.path.join(args.log_dir, 'params.txt'), 'w') as f:
        pprint.pprint(args.__dict__, f, sort_dicts=False)
        
    print('\n==> training begins')
    train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, writer, args, acc_type='top@1')
    
    #save model and close writer
    writer.close()
    print('\n==> training finished')

    


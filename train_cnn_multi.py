from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# from torchmetrics.classification import MultilabelAccuracy

import os
import random
import numpy as np

import sys
import pprint
from loaddata import *
from utils import *


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample):
#         super().__init__()
#         if downsample:
#             self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
# #                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             self.shortcut = nn.Sequential()

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
# #         self.bn1 = nn.BatchNorm2d(out_channels)
# #         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, input):
#         shortcut = self.shortcut(input)
# #         input = nn.ReLU()(self.bn1(self.conv1(input)))
# #         input = nn.ReLU()(self.bn2(self.conv2(input)))
#         input = nn.ReLU()(self.conv1(input))
#         input = nn.ReLU()(self.conv2(input))        
#         input = input + shortcut
#         return nn.ReLU()(input)


# class ResNet(nn.Module):
#     def __init__(self, in_channels, resblock, outputs=10):
#         super().__init__()

#         self.layer0 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0),
# #             nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
# #             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer1 = nn.Sequential(
#             resblock(64, 128, downsample=True),
#             resblock(128, 128, downsample=False)
#         )


#         self.layer2 = nn.Sequential(
#             resblock(128, 256, downsample=True),
#             resblock(256, 256, downsample=False)
#         )
        
#         self.layer3 = nn.Sequential(
#             resblock(256, 512, downsample=True),
#             resblock(512, 512, downsample=False)
#         )

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, outputs)
#         self.flatten = nn.Flatten() 
#     def forward(self, input):
#         input = self.layer0(input)
#         input = self.layer1(input)
#         input = self.layer2(input)
#         input = self.layer3(input)
#         input = self.gap(input)
#         input = self.flatten(input)
#         input = self.fc(input)
#         input = F.log_softmax(input, dim=1)

#         return input

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
#                 nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
#         input = nn.ReLU()(self.bn1(self.conv1(input)))
#         input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))        
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, outputs=10):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),

#             nn.MaxPool2d(kernel_size=2, stride=2, padding=1
#             nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(32, 32, downsample=False),
            resblock(32, 32, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(32, 64, downsample=True),
            resblock(64, 64, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()         

        self.fc = nn.Linear(256, outputs)
        
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, outputs)
#         self.dropout2 = nn.Dropout(0.5)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.flatten(input)
        
        input = self.fc(input)
#         input = self.fc1(input)
#         input = self.dropout2(input)
#         input = self.fc2(input)

        # input = F.log_softmax(input, dim=1) #<--- since you use BCEWithLogitsLoss

        return input


class Net(nn.Module):
    def __init__(self, feature_size_after_conv, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(feature_size_after_conv, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1) #<--- since you use BCEWithLogitsLoss
        output = x 

        return output


def train(args, model, device, train_loader, optimizer, epoch, writer):
    print(f'Epoch {epoch}:')
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if len(target.size())>1:
        #     target = torch.argmax(target, dim=1) # change from one hot to integer index

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    writer.add_scalar('Train/Loss', train_loss, epoch)
        
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # if len(target.size())>1:
            #     target = torch.argmax(target, dim=1) # change from one hot to integer index

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            
            output = torch.sigmoid(output)     #<--- since you use BCEWithLogitsLoss
            predicted = torch.round(output)
            
            correct += torch.all(torch.eq(predicted, target), dim=1).sum().item()
            # correct += (predicted == target).all().sum().item()
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    writer.add_scalar('Val/Loss', test_loss, epoch)
    writer.add_scalar('Val/Loss', test_acc, epoch)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_acc


def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--cuda', '-c', help="cuda index", type= int, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--expname', type=str, default='test')

    parser.add_argument('--output-dir', type=str, default='./results/mnist/')
    parser.add_argument('--data-dir', type=str, default= './data/')
    parser.add_argument('--restore-file', type=str, default=None, 
                        help='path to latest checkpotint')

    parser.add_argument('--model-type', type=str, default='2conv', 
                        help='model type to use (default: 2conv)')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='number of classes (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, 
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
    parser.add_argument('--print', action='store_true', help="if true, just print model info, false continue training", default=False)
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

#     parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
        
    args = parser.parse_args()
    
    # create output directory if not exists    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)
# seed for reproducibility
    def seed_torch(seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    seed_torch(args.seed)
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')

    # load model
    if args.model_type == 'resnet':
        model = ResNet(in_channels=1, resblock= ResBlock, outputs=args.num_classes).to(device)
    elif args.model_type == '2conv':
        model = Net(feature_size_after_conv=16384, num_classes=args.num_classes).to(device) # fc1 size # when, orig img size 28, 28 (9216, 128), when 36, 36 (16384, 128)
    else:
        raise NotImplementedError('model type not supported')
    
    # load dataloader
    train_loader, test_loader = fetch_dataloader(args.task, args.data_dir, device, args.batch_size, train=True)

#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True,
#                        'shuffle': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     transform=transforms.Compose([
#         transforms.ToTensor(),
# #         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('../data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    # set writer for tensorboard
    writer, current_log_path = set_writer(log_path = args.output_dir if args.restore_file is None else args.restore_file,
                        comment = args.expname, 
                        restore = args.restore_file is not None) 
    args.log_dir = current_log_path
    
    # save used param info to writer and logging directory for later retrieval
    writer.add_text('Params', pprint.pformat(args.__dict__))
    with open(os.path.join(args.log_dir, 'params.txt'), 'w') as f:
        pprint.pprint(args.__dict__, f, sort_dicts=False)
        
    # set optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # acc function
    # micro_acc_fn = MultilabelAccuracy(num_labels=args.num_classes, average='micro')
    # macro_acc_fn = MultilabelAccuracy(num_labels=args.num_classes, average='macro')
    if args.print:
        print('\n==> print model architecture')
        print(model)

        print('\n==> print model params')
        count_parameters(model)
    else:
        best_acc=0.
        path_best=None
        epoch_no_improve=0
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, writer)
            current_acc= test(model, device, test_loader, epoch, writer)
            scheduler.step()

            if (epoch%10==0) & (args.save_model):
                path_save = args.log_dir +f'/archive_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_save)

            if round(current_acc,4) > round(best_acc,4):
                best_acc= current_acc
                epoch_no_improve=0
                if path_best:
                    os.remove(path_best)
                path_best =args.log_dir +f'/best_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_best)
            else:
                epoch_no_improve+=1
            
            if epoch_no_improve >= 20:
                path_save = args.log_dir +f'/earlystop_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_save)
                status = f'===== EXPERIMENT EARLY STOPPED (no progress on val_acc for last 20 epochs) ===='
                writer.add_text('Status', status, epoch)
                print(status)
                break
                
            if epoch == args.epochs:
                torch.save(model.state_dict(), args.log_dir +f'/last_{epoch:d}_acc{val_acc:.4f}.pt')
                status = f'===== EXPERIMENT RAN TO THE END EPOCH ===='
                writer.add_text('Status', status, epoch)
                print(status)
                
        writer.close()
                
if __name__ == '__main__':
    main()
import time, math, os, sys, random
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import argparse
from utils import *
from network import *
from tensorboardX import SummaryWriter


def main(args):

    writer = SummaryWriter('logs/hdr/')

    data_dir_hdr = os.path.join(args.data_dir, "hdr")
    data_dir_ldr = os.path.join(args.data_dir, "ldr")
    log_dir = os.path.join(args.output_dir, "logs")
    im_dir = os.path.join(args.output_dir, "im")

    #=== Localize training data ===================================================

    # Get names of all images in the training path
    frames = [name for name in sorted(os.listdir(data_dir_hdr)) if os.path.isfile(os.path.join(data_dir_hdr, name))]

    # Randomize the images
    if args.rand_data:
        random.seed('111')
        random.shuffle(frames)

    # Split data into training/validation sets
    splitPos = len(frames) - 567
    frames_train, frames_valid = np.split(frames, [splitPos])

    # Number of steps per epoch depends on the number of training images
    training_samples = len(frames_train)
    validation_samples = len(frames_valid)
    steps_per_epoch = training_samples/args.batch_size

    print("\n\nData to be used:")
    print("\t%d training images" % training_samples)
    print("\t%d validation images\n" % validation_samples)
    
    train_ldr_paths = []
    train_hdr_paths = []
    train_label = []
    train_exposure = []
    for filename in frames_train:
        filename = filename.strip()
        list = filename.split('_')
#        train_ldr_paths.append(f'{data_dir_ldr}/{filename[:-4]}')
        train_ldr_paths.append(None)
        train_hdr_paths.append(f'{data_dir_hdr}/{filename}')
        train_exposure.append(float(list[1]))
        train_label.append(float(list[1]))

    valid_ldr_paths = []
    valid_hdr_paths = []
    valid_exposure = []
    valid_label = []
    for filename in frames_valid:
        filename = filename.strip()
        list = filename.split('_')
#        valid_ldr_paths.append(f'{data_dir_ldr}/{filename[:-4]}')
        train_ldr_paths.append(None)
        valid_hdr_paths.append(f'{data_dir_hdr}/{filename}')
        valid_exposure.append(float(list[1]))
        valid_label.append(float(list[1]))
    
    train_dataset = HDR2Illuminance(train_hdr_paths, train_label, args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataset = HDR2Illuminance(valid_hdr_paths, valid_label, args)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    
    
    # Network
    # net = Net().cuda()
    net = S2ConvNet_deep().cuda()
    
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # criterion_hdr = nn.MSELoss().cuda()
    criterion_reg = nn.MSELoss().cuda()


    # Optimizer
    # optim_hdr = torch.optim.Adam(net_hdr.parameters(), lr=args.lr_hdr)
    # scheduler_hdr = torch.optim.lr_scheduler.StepLR(optim_hdr, step_size=int(steps_per_epoch), gamma=0.9)
    # optim_reg = torch.optim.Adam(net_reg.parameters(), lr=args.lr_reg)
    
    total_start_time = time.time()
    for epoch in range(args.num_epochs):
        if epoch <= 500:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        elif epoch <= 800:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/10.)
        else:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/100.)

        train_loss_reg = 0
        train_count = 0
        train_correct_025 = 0
        train_correct_010 = 0
        start_time = time.time()

        for curr_iter, batch_data in enumerate(train_dataloader):
            # Turn exposure into one-hot vector, dim 20
            # exposure = torch.LongTensor(batch_data['exposure']).reshape(args.batch_size,1)
            # exposure = torch.zeros(args.batch_size,20).scatter_(1,exposure/50-1,1)
            # ldr = batch_data['ldr'].cuda()
            hdr = batch_data['hdr'].cuda()
            label = batch_data['label'].cuda() / 1000.0
            # exposure = exposure.cuda()

            net.train()
            optim.zero_grad()
            
            label_pred = net(hdr)
            label_pred = label_pred.squeeze()
            loss_reg = criterion_reg(label_pred, label.float())
            train_loss_reg += loss_reg.item()
            loss_reg.backward()
            optim.step()
            train_count += label.shape[0]
            train_correct_025 += ( (abs(label_pred-label) / label) <= 0.25 ).sum().item()
            train_correct_010 += ( (abs(label_pred-label) / label) <= 0.10 ).sum().item()
            print(curr_iter, loss_reg.item())

        writer.add_scalar('train_reg_loss', train_loss_reg / train_count, epoch)
        writer.add_scalar('train_accuracy_25%', train_correct_025 / train_count, epoch)
        writer.add_scalar('train_accuracy_10%', train_correct_010 / train_count, epoch)
        print('\rEpoch [{0}/{1}]] Train: reg loss: {2:.4f} 25%Accuracy {3:.4f} 10%Accuracy {4:.4f}'.format(
                epoch+1, args.num_epochs,
                train_loss_reg / train_count,
                train_correct_025 / train_count,
                train_correct_010 / train_count), end="")


        best_accuracy = 0
        val_loss_reg = 0
        val_count = 0
        val_correct_025 = 0
        val_correct_010 = 0
        for _, batch_data in enumerate(valid_dataloader):
            net.eval()

            # Turn exposure into one-hot vector, dim 20
            with torch.no_grad():
                hdr_val = batch_data['hdr'].cuda()
                label_val = batch_data['label'].cuda() / 1000.0
                
                label_pred_val = net(hdr_val)
                label_pred_val = label_pred_val.squeeze()
                loss_reg_val = criterion_reg(label_pred_val, label_val.float())
                val_loss_reg += loss_reg_val.item()
                val_count += label_val.shape[0]
                val_correct_025 += ( (abs(label_pred_val-label_val) / label_val) <= 0.25 ).sum().item()
                val_correct_010 += ( (abs(label_pred_val-label_val) / label_val) <= 0.10 ).sum().item()
        
        elapsed = time.time() - start_time
        
        writer.add_scalar('val_reg_loss', val_loss_reg / val_count, epoch)
        writer.add_scalar('val_accuracy_25%', val_correct_025 / val_count, epoch)
        writer.add_scalar('val_accuracy_10%', val_correct_010 / val_count, epoch)
        print('\rEpoch [{0}/{1}]] Train: reg loss: {2:.4f} 25%Accuracy {3:.4f} 10%Accuracy {3:.4f}'.format(
                epoch+1, args.num_epochs,
                val_loss_reg / val_count,
                val_correct_025 / val_count,
                val_correct_010 / val_count), end="")
        print('Epoch: %d time elapsed: %.2f hours'%(epoch+1,elapsed/3600))

        if val_correct_010 > best_accuracy:
            best_accuracy = val_correct_010
            torch.save(net, 'best-model.pt')

    total_elapsed = time.time() - total_start_time
    print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help="network architecture to use", default='deep', choices=['original', 'deep'])
    parser.add_argument("--height", default=160, type=int, help="The height of input image")
    parser.add_argument("--width", default=320, type=int, help="The width of input image")
    parser.add_argument("--data_dir",default="dataset", help='Path to processed dataset')
    parser.add_argument("--output_dir", default="training_output", help='Path to output directory, for weights and intermediate results')
    parser.add_argument("--rand_data", default=True, help='Random shuffling of training data')
    parser.add_argument("--batch_size", default=256, help='Batch size for training')
    parser.add_argument('--train_size', default=0.99, help='Fraction of data to use for training, the rest is validataion data')
    parser.add_argument("--num_epochs", default=1000, help='Number of training epochs')
    parser.add_argument("--lr", default=1e-3, help='Learning rate of HDR reconstruction network')
    # parser.add_argument("--lr_reg", default=5e-3, help='Learning rate of spherical regression network')
    parser.add_argument("--num_workers", default=0, help='Number of workers')
#    parser.add_argument("--hdr", default=True, help='Whether or not include illuminance loss')
    parser.add_argument("--reg_only", default=True, help='Whether or not include illuminance loss')

    parser.add_argument("--bandwidth", default=30, type=int, help="the bandwidth of the S2 signal", required=False)
                        
                        
                        

    
    args = parser.parse_args()



    main(args)

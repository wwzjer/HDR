import time, math, os, sys, random
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import argparse
from utils import *
from network import *
from tensorboardX import SummaryWriter


def main(args):

    exp_name = 'hdr2illuminance'
    writer = SummaryWriter(f'logs/{exp_name}/')
    
    eps = 1.0/255.0

    data_dir_hdr = os.path.join(args.data_dir, "hdr160320")
    data_dir_ldr = os.path.join(args.data_dir, "ldr160320")
    log_dir = os.path.join(args.output_dir, "logs")
    im_dir = os.path.join(args.output_dir, "im")

    #=== Localize training data ===================================================

    # Get names of all images in the training path
    frames = [name for name in sorted(os.listdir(data_dir_hdr)) if os.path.isfile(os.path.join(data_dir_hdr, name))]
    # total 3657: train 2500, val 500, test 657
    # Randomize the images
    if args.rand_data:
        random.seed('111')
        random.shuffle(frames)
    
    frames = frames[:3500] # the rest are for testing

    # Split data into training/validation sets
    splitPos = len(frames) - 500
    frames_train, frames_valid = np.split(frames, [splitPos])

    # Number of steps per epoch depends on the number of training images
    training_samples = len(frames_train)
    validation_samples = len(frames_valid)

    print("\n\nData to be used:")
    print("\t%d training HDRs" % training_samples)
    print("\t%d validation HDRs\n" % validation_samples)
    
    train_hdr_paths = []
    train_label = []
    brackets = list(range(50,1050,50))
    for filename in frames_train:
        filename = filename.strip()
        namelist = filename.split('_')
        train_hdr_paths.append(f'{data_dir_hdr}/{filename}')
        train_label.append(float(namelist[3][:-4]))

    valid_hdr_paths = []
    valid_label = []
    
    for filename in frames_valid:
        filename = filename.strip()
        namelist = filename.split('_')
        valid_hdr_paths.append(f'{data_dir_hdr}/{filename}')
        valid_label.append(float(namelist[3][:-4]))
    
    train_dataset = HDR2Illuminance(train_hdr_paths, train_label, args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valid_dataset = HDR2Illuminance(valid_hdr_paths, valid_label, args)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Network
    
    # if args.hdr:
    #     net_hdr = HDR_net().cuda()
    # if args.reg:
    #     net_reg = S2ConvNet_deep().cuda()
    # else:
    #     net_reg = torch.load('best-model.pt')
    if args.restore:
        state = torch.load('checkpoint/end2end_exposure_method2_16-best-model.pth')
        net = state['net']
        cur_epoch = state['epoch']
    else:
        net = S2ConvNet_deep(args).cuda()
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        cur_epoch = 0
    
#    s2cnn = S2ConvNet_deep().cuda()
#    img = cv.imread('dataset/hdr/0706_126_0_1330.hdr',-1)[:,:,::-1]*255.0
#    img = cv.resize(img, (320,160), interpolation=cv.INTER_CUBIC)
#    img = img.transpose(2,0,1) / 255.0
#    img = torch.Tensor(img).cuda()
#    out1 = s2cnn.projection(img.unsqueeze(0))
#    import matplotlib.pyplot as plt
#    plt.imsave('out_new.png', out1.squeeze(0).squeeze(0).cpu().numpy())

    
#    torch.distributed.init_process_group(backend="nccl")
#    net = torch.nn.parallel.DistributedDataParallel(net)


    criterion_hdr = nn.MSELoss().cuda()
    criterion_reg = nn.MSELoss().cuda()


    # Optimizer
    # optim_hdr = torch.optim.Adam(net_hdr.parameters(), lr=args.lr_hdr)
    # scheduler_hdr = torch.optim.lr_scheduler.StepLR(optim_hdr, step_size=int(steps_per_epoch), gamma=0.9)
    # optim_reg = torch.optim.Adam(net_reg.parameters(), lr=args.lr_reg)
    
    total_start_time = time.time()
    for epoch in range(cur_epoch, args.num_epochs):
        if epoch <= 50:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        elif epoch <= 80:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/10.)
        else:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/100.)

        train_loss = 0
        train_loss_reg = 0
        train_loss_hdr = 0
        train_count = 0
        train_correct_025 = 0
        train_correct_010 = 0
        start_time = time.time()

        for curr_iter, batch_data in enumerate(train_dataloader):
            # Turn exposure into one-hot vector, dim 20
            hdr = batch_data['hdr'].cuda()
            label = batch_data['label'].cuda() / 1000.0

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
            if curr_iter % 100 == 0:
                print(epoch, curr_iter, loss_reg.item())
                print('time: ', time.time() - start_time)

        writer.add_scalar('train_reg_loss', train_loss_reg / train_count, epoch)
        writer.add_scalar('train_accuracy_25%', train_correct_025 / train_count, epoch)
        writer.add_scalar('train_accuracy_10%', train_correct_010 / train_count, epoch)
        print('Epoch [{0}/{1}]] Train: reg loss: {2:.2f} 25%Accuracy {3:.2f} 10%Accuracy {4:.2f}'.format(
                epoch+1, args.num_epochs,
                train_loss_reg / train_count,
                train_correct_025 / train_count,
                train_correct_010 / train_count), end="")


        best_accuracy = 0
        val_loss = 0
        val_loss_reg = 0
        val_loss_hdr = 0
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
        
        val_accuracy_025 = val_correct_025 / val_count
        val_accuracy_010 = val_correct_010 / val_count
        
        writer.add_scalar('val_reg_loss', val_loss_reg / val_count, epoch)
        writer.add_scalar('val_accuracy_25%', val_accuracy_025, epoch)
        writer.add_scalar('val_accuracy_10%', val_accuracy_010, epoch)
        print('\nEpoch [{0}/{1}]] Valid: reg loss: {2:.2f} 25%Accuracy {3:.2f} 10%Accuracy {4:.2f}'.format(
                epoch+1, args.num_epochs,
                val_loss_reg / val_count,
                val_correct_025 / val_count,
                val_correct_010 / val_count), end="")
        elapsed = time.time() - start_time
        print('\nEpoch: %d time elapsed: %.2f hours'%(epoch+1,elapsed/3600))

        
        if val_accuracy_010 > best_accuracy:
            best_accuracy = val_accuracy_010
            state = {
                'net': net,
                'epoch': epoch,
            }
            
            torch.save(state, f'checkpoint/{exp_name}-best-model.pth')

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
    parser.add_argument("--batch_size", default=64, help='Batch size for training')

    parser.add_argument("--num_epochs", default=100, help='Number of training epochs')
    parser.add_argument("--lr", default=1e-4, help='Learning rate of HDR reconstruction network')
    # parser.add_argument("--lr_reg", default=5e-3, help='Learning rate of spherical regression network')
    parser.add_argument("--num_workers", default=4, help='Number of workers')
    parser.add_argument("--hdr", default=True, help='Whether or not include illuminance loss')
    parser.add_argument("--reg", default=False, help='Whether or not include illuminance loss')

    parser.add_argument("--bandwidth", default=30, type=int, help="the bandwidth of the S2 signal", required=False)
    parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
    parser.add_argument("--adding_exposure", default=False, type=bool, help="Whether or not encode exposure information")
    parser.add_argument("--restore", default=False, type=bool, help="")
                        
                        
                        

    
    args = parser.parse_args()



    main(args)

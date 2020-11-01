import time, math, os, sys, random
import cv2 as cv
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import argparse
from utils import *
from network_integrate import *
from tensorboardX import SummaryWriter


def activate_sync_bn(m):
    if isinstance(m, torch.nn.modules.SyncBatchNorm):
        m._specify_ddp_gpu_num(1)
        
def set_bn_eval(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        m.eval()

def main(args):
    
    eps = 1.0/255.0
    
    if args.integral:
        if args.adding_exposure:
            exp_name = 'scaling_wIntegral_wExposure'
        else:
            exp_name = 'scaling_wIntegrate_woExposure'
    else:
        if args.adding_exposure:
            exp_name = 'scaling_woIntegrate_wExposure'
        else:
            exp_name = 'scaling_woIntegrate_woExposure'
    
    if args.pretrain:
        exp_name = 'finetune_wIntegral_wExposure'
    
    print(exp_name)
        
    writer = SummaryWriter(f'logs/{exp_name}/')
        
    log_dir = os.path.join(args.output_dir, "logs")
    
    if not args.diff_domain:
       # Training and testing both from store '0081' and '0126'
        data_dir_hdr = [os.path.join(args.data_dir, "multi-stores/0081_hdr"),
                              os.path.join(args.data_dir, "multi-stores/0126_hdr")]
        data_dir_ldr = [os.path.join(args.data_dir, "multi-stores/0081_ldr"),
                              os.path.join(args.data_dir, "multi-stores/0126_ldr")]
    else:
        # training and validation/testing in different scenes.
        data_dir_hdr_train = [os.path.join(args.data_dir, "multi-stores/0081_hdr"),
                              os.path.join(args.data_dir, "multi-stores/0126_hdr"),
                              os.path.join(args.data_dir, "multi-stores/0055_hdr"),
                              os.path.join(args.data_dir, "multi-stores/0108_hdr"),
                              os.path.join(args.data_dir, "multi-stores/0163_hdr")]
        data_dir_hdr_valid = os.path.join(args.data_dir, "multi-stores/0058_hdr")
        data_dir_ldr_train = [os.path.join(args.data_dir, "multi-stores/0081_ldr"),
                              os.path.join(args.data_dir, "multi-stores/0126_ldr"),
                              os.path.join(args.data_dir, "multi-stores/0055_ldr"),
                              os.path.join(args.data_dir, "multi-stores/0108_ldr"),
                              os.path.join(args.data_dir, "multi-stores/0163_ldr")]
        data_dir_ldr_valid = os.path.join(args.data_dir, "multi-stores/0058_ldr")

    #=== Localize training data ===================================================

    # Get names of all images in the training path
    frames = []
    frames_train = []
    frames_valid = []
    if args.diff_domain:
        for path in data_dir_hdr_train:
            frames_train += [os.path.join(path, name) for name in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, name))]
        frames_valid = [os.path.join(data_dir_hdr_valid, name) for name in sorted(os.listdir(data_dir_hdr_valid)) if os.path.isfile(os.path.join(data_dir_hdr_valid, name))]
    else:
        for path in data_dir_hdr:
            frame += [os.path.join(path, name) for name in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, name))]
        if args.rand_data:
            random.seed('111')
            random.shuffle(frames)
        frames = frames[:3157] # the rest are for testing
        # Split data into training/validation sets
        splitPos = len(frames) - 157
        frames_train, frames_valid = np.split(frames, [splitPos])

    frames_train = frames_train[:10]
    frames_valid = frames_valid[:10]
    
    # Number of steps per epoch depends on the number of training images
    training_samples = len(frames_train)
    validation_samples = len(frames_valid)

    print("\n\nData to be used:")
    print("\t%d training HDRs" % training_samples)
    print("\t%d validation HDRs\n" % validation_samples)
    
    train_ldr_paths = []
    train_hdr_paths = []
    train_exposure = []
    train_label = []
    train_storeid = []
    brackets = list(range(50,1050,50))
    for filename in frames_train:
        filename = filename.strip()
        namelist = filename.split('_')
        for name in os.listdir(filename[:26] + 'l' + filename[27:-4]):
            train_storeid.append(filename[21:25])
            train_hdr_paths.append(filename)
            ldr_name = filename[:26] + 'l' + filename[27:-4] + '/' + name
            train_ldr_paths.append(ldr_name)
#            bb = random.choice(brackets)
            train_exposure.append(ldr_name.split('_')[-2])
            train_label.append(float(namelist[-1][:-4]))

    valid_ldr_paths = []
    valid_hdr_paths = []
    valid_exposure = []
    valid_label = []
    valid_storeid = []
    for filename in frames_valid:
        filename = filename.strip()
        namelist = filename.split('_')
        for name in os.listdir(filename[:26] + 'l' + filename[27:-4]):
            valid_storeid.append(filename[21:25])
            valid_hdr_paths.append(filename)
            ldr_name = filename[:26] + 'l' + filename[27:-4] + '/' + name
            valid_ldr_paths.append(ldr_name)
#            bb = random.choice(brackets)
            valid_exposure.append(ldr_name.split('_')[-2])
            valid_label.append(float(namelist[-1][:-4]))
    
    train_dataset = PairwiseDataset(train_ldr_paths, train_hdr_paths, train_exposure, train_label, train_storeid, args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valid_dataset = PairwiseDataset(valid_ldr_paths, valid_hdr_paths, valid_exposure, valid_label, valid_storeid, args)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Network
    net = Net(args).cuda()
#    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
#    net.apply(activate_sync_bn)
    if torch.cuda.device_count() > 1:
#        net = nn.DataParallel(net)
        torch.distributed.init_process_group(backend="nccl")
        net = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False)
    cur_epoch = 0
    if args.pretrain == True:
        if args.adding_exposure:
            pretrain_name = 'pretrain_wExposure'
        else:
            pretrain_name = 'pretrain_woExposure'
        checkpoint = torch.load(f'checkpoint/{pretrain_name}_best-model.pth.tar')
        net.load_state_dict(checkpoint['net'])
        cur_epoch = 0
    else:
        if args.restore:
            checkpoint = torch.load(f'checkpoint/{exp_name}_best-model.pth.tar')
            net.load_state_dict(checkpoint['net'])
            cur_epoch = checkpoint['epoch']
            
    vgg = torchvision.models.__dict__['vgg16'](pretrained=True).cuda()
    if torch.cuda.device_count() > 1:
#        vgg = nn.DataParallel(vgg)
#        vgg_feature = vgg.module.features
        vgg = torch.nn.parallel.DistributedDataParallel(vgg, broadcast_buffers=False)
        vgg_feature = vgg.module.features
    else:
        vgg_feature = vgg.module.features
    
    criterion = nn.MSELoss().cuda()
#    criterion_reg = nn.MSELoss().cuda()
#    criterion_exp = nn.MSELoss().cuda()
    
    total_start_time = time.time()
    for epoch in range(cur_epoch, args.num_epochs):
        if epoch <= 5:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        elif epoch <= 8:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/5.)
        else:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr/10.)

        train_loss = 0
        train_loss_reg = 0
        train_loss_hdr = 0
        train_loss_tv = 0
        train_loss_p = 0
#        train_loss_consistency = 0
        train_count = 0
        train_correct_025 = 0
        train_correct_010 = 0
        start_time = time.time()

        for curr_iter, batch_data in enumerate(train_dataloader):
            # Turn exposure into one-hot vector, dim 20
            ldr = batch_data['ldr'][0].cuda()
            ldr_1 = batch_data['ldr'][1].cuda()
            hdr = batch_data['hdr'].cuda()
            exposure = torch.LongTensor(batch_data['exposure'][0]).reshape(ldr.shape[0],1)
            exposure = torch.zeros(args.batch_size,20).scatter_(1,exposure/50-1,1)
            exposure = exposure.cuda()
            exposure_1 = torch.LongTensor(batch_data['exposure'][1]).reshape(ldr.shape[0],1)
            exposure_1 = torch.zeros(args.batch_size,20).scatter_(1,exposure_1/50-1,1)
            exposure_1 = exposure_1.cuda()
            label = batch_data['label'].cuda() / 1000.0
            
            net.train()
            
            ## Fix batchNorm stats from pre-trained models
                
            net.apply(set_bn_eval)
                
            optim.zero_grad()

            hdr_pred, label_pred = net(ldr, exposure)
#            hdr_pred_1, _ = net(ldr_1, exposure_1)
            # HDR reconstruction loss
            loss_hdr = criterion(torch.log(hdr_pred+eps), torch.log(hdr+eps))
            train_loss_hdr += loss_hdr.item()
            # HDR prediction TV loss
            loss_tv = tv_loss(hdr_pred)
            train_loss_tv += loss_tv.item()
            # VGG Perceptual loss
            loss_p = perceptual_loss(vgg_feature, hdr_pred, hdr)
            train_loss_p += loss_p.item()
            # HDR prediction consistency loss
#            loss_consistency = criterion(hdr_pred, hdr_pred_1)
#            train_loss_consistency += loss_consistency.item()
            # Illuminance estimation loss
            label_pred = label_pred.squeeze()
            loss_reg = criterion(label_pred, label.float())
            train_loss_reg += loss_reg.item()
            # ==== Total loss ==================================
            if args.integral:
                loss = loss_hdr + loss_reg + 0.1 * loss_tv + 0.001 * loss_p
            else:
                loss = loss_hdr + 0.1 * loss_tv + 0.001 * loss_p
            loss.backward()
            optim.step()
            train_loss += loss.item()
            train_count += ldr.shape[0]
            train_correct_025 += ( (abs(label_pred-label) / label) <= 0.25 ).sum().item()
            train_correct_010 += ( (abs(label_pred-label) / label) <= 0.10 ).sum().item()
            if curr_iter % 100 == 0:
                print(epoch, curr_iter, loss.item(), loss_hdr.item(), loss_reg.item(), loss_tv.item(), loss_p.item())
                print('time: ', time.time() - start_time)

        writer.add_scalar('train_loss', train_loss / train_count, epoch)
        writer.add_scalar('train_hdr_loss', train_loss_hdr / train_count, epoch)
        writer.add_scalar('train_reg_loss', train_loss_reg / train_count, epoch)
        writer.add_scalar('train_tv_loss', train_loss_tv / train_count, epoch)
        writer.add_scalar('train_perceptual_loss', train_loss_p / train_count, epoch)
#        writer.add_scalar('train_consistency_loss', train_loss_consistency / train_count, epoch)
        writer.add_scalar('train_accuracy_25%', train_correct_025 / train_count, epoch)
        writer.add_scalar('train_accuracy_10%', train_correct_010 / train_count, epoch)
        print('Epoch [{0}/{1}]] Train: total loss: {2:.2f} hdr loss: {3:.2f} reg loss: {4:.2f} TV loss: {5:.2f} Perceptual loss: {6:.2f} 25%Accuracy {7:.2f} 10%Accuracy {8:.2f}'.format(
                epoch+1, args.num_epochs,
                train_loss / train_count,
                train_loss_hdr / train_count,
                train_loss_reg / train_count,
                train_loss_tv / train_count,
                train_loss_p / train_count,
#                train_loss_consistency / train_count,
                train_correct_025 / train_count,
                train_correct_010 / train_count), end="\n")

        best_accuracy = 0
        val_loss = 0
        val_loss_reg = 0
        val_loss_hdr = 0
        val_loss_p = 0
        val_loss_tv = 0
#        val_loss_consistency = 0
        val_count = 0
        val_correct_025 = 0
        val_correct_010 = 0
        for curr_iter, batch_data in enumerate(valid_dataloader):
            net.eval()

            # Turn exposure into one-hot vector, dim 20
            with torch.no_grad():
                ldr_val = batch_data['ldr'][0].cuda()
                ldr_val_1 = batch_data['ldr'][1].cuda()
                hdr_val = batch_data['hdr'].cuda()
                label_val = batch_data['label'].cuda() / 1000.0
                exposure_val = torch.LongTensor(batch_data['exposure'][0]).reshape(ldr_val.shape[0],1)
                exposure_val = torch.zeros(args.batch_size,20).scatter_(1,exposure_val/50-1,1)
                exposure_val = exposure_val.cuda()
                exposure_val_1 = torch.LongTensor(batch_data['exposure'][1]).reshape(ldr_val.shape[0],1)
                exposure_val_1 = torch.zeros(args.batch_size,20).scatter_(1,exposure_val_1/50-1,1)
                exposure_val_1 = exposure_val_1.cuda()
                
                hdr_pred_val, label_pred_val = net(ldr_val, exposure_val)
#                hdr_pred_val_1, _ = net(ldr_val_1, exposure_val_1)
                
                # HDR reconstruction loss
                loss_hdr_val = criterion(torch.log(hdr_pred_val+eps), torch.log(hdr_val+eps))
                val_loss_hdr += loss_hdr_val.item()
                # HDR prediction TV loss
                loss_tv_val = tv_loss(hdr_pred_val)
                val_loss_tv += loss_tv_val.item()
                # VGG Perceptual loss
                loss_p_val = perceptual_loss(vgg_feature, hdr_pred_val, hdr_val)
                val_loss_p += loss_p_val.item()
                # HDR prediction consistency loss
#                loss_consistency_val = criterion(hdr_pred_val, hdr_pred_val_1)
#                val_loss_consistency += loss_consistency.item()
                # Illuminance estimation loss
                label_pred_val = label_pred_val.squeeze()
                loss_reg_val = criterion(label_pred_val, label_val.float())
                val_loss_reg += loss_reg_val.item()
                # ==== Total loss ==================================
                if args.integral:
                    loss_val = loss_hdr_val + loss_reg_val + 0.01 * loss_tv_val + 0.001 * loss_p_val
                else:
                    loss_val = loss_hdr_val + 0.1 * loss_tv_val + 0.001 * loss_p_val
                val_loss += loss_val.item()
                val_count += ldr_val.shape[0]
                val_correct_025 += ( (abs(label_pred_val-label_val) / label_val) <= 0.25 ).sum().item()
                val_correct_010 += ( (abs(label_pred_val-label_val) / label_val) <= 0.10 ).sum().item()
                    
                if curr_iter % 100 == 0:
                    print(curr_iter, loss.item(), loss_hdr_val.item(), loss_reg_val.item(), loss_tv_val.item(), loss_p_val.item())
        
        val_accuracy_010 = val_correct_010 / val_count
        val_accuracy_025 = val_correct_025 / val_count
        
        writer.add_scalar('val_loss', val_loss / val_count, epoch)
        writer.add_scalar('val_hdr_loss', val_loss_hdr / val_count, epoch)
        writer.add_scalar('val_reg_loss', val_loss_reg / val_count, epoch)
        writer.add_scalar('val_tv_loss', val_loss_tv / val_count, epoch)
        writer.add_scalar('val_perceptual_loss', val_loss_p / val_count, epoch)
#        writer.add_scalar('val_consistency_loss', val_loss_consistency / val_count, epoch)
        writer.add_scalar('val_accuracy_25%', val_accuracy_025, epoch)
        writer.add_scalar('val_accuracy_10%', val_accuracy_010, epoch)
        print('\nEpoch [{0}/{1}]] Valid: total loss: {2:.2f} hdr loss: {3:.2f} reg loss: {4:.2f} tv loss: {5:.2f} perceptual loss: {6:.2f} 25%Accuracy {7:.2f} 10%Accuracy {8:.2f}'.format(
                epoch+1, args.num_epochs,
                val_loss / val_count,
                val_loss_hdr / val_count,
                val_loss_reg / val_count,
                val_loss_tv / val_count,
                val_loss_p / val_count,
#                val_loss_consistency / val_count,
                val_correct_025 / val_count,
                val_correct_010 / val_count), end="\n")
        elapsed = time.time() - start_time
        print('\nEpoch: %d time elapsed: %.2f hours'%(epoch+1,elapsed/3600))

        
        if val_accuracy_010 > best_accuracy:
            best_accuracy = val_accuracy_010
            state = {
                'net': net,
                'epoch': epoch,
            }
            torch.save(state, f'checkpoint/{exp_name}_best-model.pth')

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
    parser.add_argument("--batch_size", default=16, help='Batch size for training')
    parser.add_argument("--num_epochs", default=10, help='Number of training epochs')
    parser.add_argument("--lr", default=1e-4, help='Learning rate of HDR reconstruction network')
    # parser.add_argument("--lr_reg", default=5e-3, help='Learning rate of spherical regression network')
    parser.add_argument("--num_workers", default=4, help='Number of workers')
#    parser.add_argument("--hdr", default=True, help='Whether or not include illuminance loss')
#    parser.add_argument("--reg", default=False, help='Whether or not include illuminance loss')

#    parser.add_argument("--bandwidth", default=30, type=int, help="the bandwidth of the S2 signal", required=False)
    parser.add_argument("--local_rank", default=1, type=int, help="local_rank")
    parser.add_argument("--restore", default=False, type=bool, help="")
    parser.add_argument("--integral", default=True, type=bool, help="Whether of not adding integral loss")
    parser.add_argument("--adding_exposure", default=True, type=bool, help="Whether or not encode exposure information")
    parser.add_argument("--diff_domain", default=True, type=bool, help="Whether or not training and testing in different domains")
    parser.add_argument("--pretrain", default=False, type=bool, help="Whether or not training and testing in different domains")
                        
    args = parser.parse_args()

    main(args)

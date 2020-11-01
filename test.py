import time, math, os, sys, random
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
from torch.autograd import Variable
import argparse
from utils import *
from network_integrate import *
from tensorboardX import SummaryWriter



#def activate_sync_bn(m):
#    if isinstance(m, torch.nn.modules.SyncBatchNorm):
#        m._specify_ddp_gpu_num(1)




def main(args):
    
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
    
    print(exp_name)
    
    
    net = Net(args).cuda()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
#        torch.distributed.init_process_group(backend="nccl")
#        net = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False)
#    torch.distributed.init_process_group(backend="nccl")
    net = torch.load(f'checkpoint/{exp_name}_best-model.pth')['net']
    
    
    
    vgg =  torchvision.models.__dict__['vgg16'](pretrained=True).cuda()
    if torch.cuda.device_count() > 1:
        vgg = nn.DataParallel(vgg)
#        vgg = torch.nn.parallel.DistributedDataParallel(vgg, broadcast_buffers=False)
        vgg_feature = vgg.module.features
    else:
        vgg_feature = vgg.features
        
    eps = 1.0/255.0

    if not args.diff_domain:
       # Training and testing both from store '0081' and '0126'
        data_dir_hdr = [os.path.join(args.data_dir, "multi-stores/0081_hdr"),
                        os.path.join(args.data_dir, "multi-stores/0126_hdr")]
        data_dir_ldr = [os.path.join(args.data_dir, "multi-stores/0081_ldr"),
                        os.path.join(args.data_dir, "multi-stores/0126_ldr")]
    else:
        # training and validation/testing in different scenes.
        data_dir_hdr_test = [os.path.join(args.data_dir, "multi-stores/1028_hdr"),
                             os.path.join(args.data_dir, "multi-stores/1055_hdr")]
#        data_dir_hdr_test = [os.path.join(args.data_dir, "multi-stores/0081_hdr"),
#                                os.path.join(args.data_dir, "multi-stores/0126_hdr"),
#                                os.path.join(args.data_dir, "multi-stores/0055_hdr"),
#                                os.path.join(args.data_dir, "multi-stores/0108_hdr"),
#                                os.path.join(args.data_dir, "multi-stores/0163_hdr")]
        data_dir_ldr_test = [os.path.join(args.data_dir, "multi-stores/1028_ldr"),
                             os.path.join(args.data_dir, "multi-stores/1055_ldr")]
#        data_dir_ldr_test = [os.path.join(args.data_dir, "multi-stores/0081_ldr"),
#                                os.path.join(args.data_dir, "multi-stores/0126_ldr"),
#                                os.path.join(args.data_dir, "multi-stores/0055_ldr"),
#                                os.path.join(args.data_dir, "multi-stores/0108_ldr"),
#                                os.path.join(args.data_dir, "multi-stores/0163_ldr")]
        
    # Get names of all images in the training path
    frames = []
    frames_test = []
    if args.diff_domain:
        for path in data_dir_hdr_test:
            frames_test += [os.path.join(path, name) for name in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, name))]
    else:
        for path in data_dir_hdr:
            frame += [os.path.join(path, name) for name in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, name))]
        if args.rand_data:
            random.seed('111')
            random.shuffle(frames)
        frames_test = frames[3257:] # the rest are for testing
    
    
#    frames_test = frames_test[:10]
    
    testing_samples = len(frames_test)

    print("\n\nData to be used:")
    print("\t%d testing HDRs" % testing_samples)

    
    test_ldr_paths = []
    test_hdr_paths = []
    test_exposure = []
    test_label = []
    test_storeid = []
    brackets = list(range(50,1050,50))
    for filename in frames_test:
        filename = filename.strip()
        namelist = filename.split('_')
        for name in os.listdir(filename[:26] + 'l' + filename[27:-4]):
            test_storeid.append(filename[21:25])
            test_hdr_paths.append(filename)
            ldr_name = filename[:26] + 'l' + filename[27:-4] + '/' + name
            test_ldr_paths.append(ldr_name)
#            bb = random.choice(brackets)
            test_exposure.append(ldr_name.split('_')[-2])
            test_label.append(float(namelist[-1][:-4]))

    test_dataset = PairwiseDataset(test_ldr_paths, test_hdr_paths, test_exposure, test_label, test_storeid, args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)



#    dphi = 2 * np.pi/args.width
#    dtheta = np.pi/args.height
#
#    phi = torch.zeros((args.height//2, args.width))
#    theta = torch.zeros((args.height//2, args.width))
#
#    for y in range(args.height//2):
#        for x in range(args.width):
#            phi[y,x] = (x / args.width) * 2 * np.pi
#            theta[y,x] = (y / args.height) * np.pi
#    phi = phi.unsqueeze(0).repeat(args.batch_size,1,1).cuda()
#    theta = theta.unsqueeze(0).repeat(args.batch_size,1,1).cuda()

    criterion = nn.MSELoss().cuda()


    test_loss = 0
    test_loss_reg = 0
    test_loss_hdr = 0
    test_loss_tv = 0
    test_loss_p = 0
#    test_loss_consistency = 0
    test_count = 0
    test_correct_025 = 0
    test_correct_010 = 0
    for curr_iter, batch_data in enumerate(test_dataloader):
        net.eval()
        with torch.no_grad():
            ldr = batch_data['ldr'][0].cuda()
#            ldr_1 = batch_data['ldr'][1].cuda()
            hdr = batch_data['hdr'].cuda()
            exposure = torch.LongTensor(batch_data['exposure'][0]).reshape(ldr.shape[0],1)
            exposure = torch.zeros(args.batch_size,20).scatter_(1,exposure/50-1,1)
            exposure = exposure.cuda()
#            exposure_1 = torch.LongTensor(batch_data['exposure'][1]).reshape(ldr.shape[0],1)
#            exposure_1 = torch.zeros(args.batch_size,20).scatter_(1,exposure_1/50-1,1)
#            exposure_1 = exposure_1.cuda()
            label = batch_data['label'].cuda() / 1000.0
        
        
            hdr_pred, label_pred = net(ldr, exposure)
#            hdr_pred_1, _ = net(ldr_1, exposure_1)
            label_pred = label_pred.squeeze()
#            for i in range(16):
#                cv.imwrite(f'test{i}.hdr', hdr_pred_test[i,:,:,:].cpu().numpy().transpose(1,2,0)[:,:,::-1])
#                cv.imwrite(f'gt{i}.hdr', hdr_test[i,:,:,:].cpu().numpy().transpose(1,2,0)[:,:,::-1])
#            img = 179 * (hdr_pred_test[:,0,:,:] * 0.2126 + hdr_pred_test[:,1,:,:] * 0.7152 + hdr_pred_test[:,2,:,:] * 0.0722)
#            output = img[:,0:args.height//2,:].mul(torch.sin(theta)).mul(torch.cos(theta)) * dphi * dtheta
#
#            label_pred_test = output.sum([1,2]) / 1000.0
            
            # HDR reconstruction loss
            loss_hdr = criterion(torch.log(hdr_pred+eps), torch.log(hdr+eps))
            test_loss_hdr += loss_hdr.item()
            # HDR prediction TV loss
            loss_tv = tv_loss(hdr_pred)
            test_loss_tv += loss_tv.item()
            # VGG Perceptual loss
            loss_p = perceptual_loss(vgg_feature, hdr_pred, hdr)
            test_loss_p += loss_p.item()
            # HDR prediction consistency loss
#            loss_consistency = criterion(hdr_pred, hdr_pred_1)
#            test_loss_consistency += loss_consistency.item()
            # Illuminance estimation loss
            label_pred = label_pred.squeeze()
            loss_reg = criterion(label_pred, label.float())
            test_loss_reg += loss_reg.item()
            # ==== Total loss ==================================
            if args.integral:
                loss = loss_hdr + loss_reg + 0.1 * loss_tv + 0.001 * loss_p
            else:
                loss = loss_hdr + 0.1 * loss_tv + 0.001 * loss_p
            test_loss += loss.item()
            
            test_count += label.shape[0]
            test_correct_025 += ( (abs(label_pred-label) / label) <= 0.25 ).sum().item()
            test_correct_010 += ( (abs(label_pred-label) / label) <= 0.10 ).sum().item()

            if curr_iter % 100 == 0:
#                print(curr_iter, test_loss.item(), test_loss_hdr.item(), test_loss_reg.item(), test_loss_tv.item(), test_loss_p.item())
                print(curr_iter, '25% Accuracy', ( (abs(label_pred-label) / label) <= 0.25 ).sum().item()/label.shape[0])
                print(curr_iter, '10% Accuracy', ( (abs(label_pred-label) / label) <= 0.10 ).sum().item()/label.shape[0])

    print('\n test: total loss: {0:.2f} hdr loss: {1:.2f} reg loss: {2:.2f} tv loss: {3:.2f} perceptual loss: {4:.2f} 25%Accuracy {5:.4f} 10%Accuracy {6:.4f}'.format(
    test_loss / test_count,
    test_loss_hdr / test_count,
    test_loss_reg / test_count,
    test_loss_tv / test_count,
    test_loss_p / test_count,
#    test_loss_consistency / test_count,
    test_correct_025 / test_count,
    test_correct_010 / test_count))

    
#####################################################################################################################
#    path = 'dataset/ldr160320_sample/'
#    batch_img = []
#    batch_exposure = []
#    bracket = list(range(50,1050,50))
#    for i in range(1,20):
#        print(f'{path}/{i}.png')
#        img = cv.imread(f'{path}/{i}.png',-1)[:,:,::-1]
##        gt = cv.imread('dataset/hdr/0706_126_0_1330.hdr',-1)[:,:,::-1]
#        img = cv.resize(img, (args.width, args.height), interpolation=cv.INTER_CUBIC)
#        img = img.transpose(2,0,1) / 255.0
#        img = torch.Tensor(img)
#        batch_img.append(img.unsqueeze(0))
#        exposure = bracket[i]
#        batch_exposure.append(exposure)
#
#    batch_img = torch.cat(batch_img, dim=0).cuda()
#    batch_exposure = torch.Tensor(batch_exposure).unsqueeze(1).cuda()
#    print(batch_img.shape)
#    print(batch_exposure.shape)
#    batch_img = torch.Tensor(batch_img).unsqueeze(0).cuda()
#
#
#    with torch.no_grad():
#        hdr_pred, label_pred = net(batch_img, batch_exposure)
#
#    for i in range(19):
#        print(hdr_pred[i+1,:,:,:].shape)
#        cv.imwrite(f'{path}/hdr_pred_{i+1}.hdr', hdr_pred[i,:,:,:].cpu().numpy().transpose(1,2,0)[:,:,::-1])
#
#
##    cv.imwrite('ldr.jpg', img.cpu().squeeze(0).numpy().transpose(1,2,0) * 255.0)
##    cv.imwrite('gt.hdr', gt)
#
#    import matplotlib.pyplot as plt
#    plt.imsave('out_new.png', out1.squeeze(0).squeeze(0).cpu().numpy())

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help="network architecture to use", default='deep', choices=['original', 'deep'])
    parser.add_argument("--height", default=160, type=int, help="The height of input image")
    parser.add_argument("--width", default=320, type=int, help="The width of input image")
    parser.add_argument("--data_dir",default="dataset", help='Path to processed dataset')
    parser.add_argument("--output_dir", default="training_output", help='Path to output directory, for weights and intermediate results')
    parser.add_argument("--rand_data", default=True, help='Random shuffling of training data')
    parser.add_argument("--batch_size", default=16, help='Batch size for training')
    parser.add_argument("--num_epochs", default=100, help='Number of training epochs')
    parser.add_argument("--lr", default=1e-4, help='Learning rate of HDR reconstruction network')
    # parser.add_argument("--lr_reg", default=5e-3, help='Learning rate of spherical regression network')
    parser.add_argument("--num_workers", default=4, help='Number of workers')
#    parser.add_argument("--hdr", default=True, help='Whether or not include illuminance loss')
#    parser.add_argument("--reg", default=False, help='Whether or not include illuminance loss')

    parser.add_argument("--bandwidth", default=30, type=int, help="the bandwidth of the S2 signal", required=False)
    parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
    parser.add_argument("--integral", default=False, type=bool, help="Whether of not adding integral loss")
    parser.add_argument("--adding_exposure", default=True, type=bool, help="Whether or not encode exposure information")
    parser.add_argument("--diff_domain", default=True, type=bool, help="Whether or not training and testing in different domains")
                        
    args = parser.parse_args()

    main(args)




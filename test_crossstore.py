import time, math, os, sys, random
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import argparse
from utils import *
from network_integrate import *
from tensorboardX import SummaryWriter


def main(args):
    
    net = torch.load('checkpoint/scaling_wIntegral_wExposure_best-model.pth')['net']
    
    
    eps = 1.0/255.0

#    data_dir_hdr = os.path.join(args.data_dir, "hdr160320")
    data_dir_ldr = os.path.join(args.data_dir, "imgs_101000055")
    data_dir_ldr = os.path.join(args.data_dir, "multi-stores_160320")
    log_dir = os.path.join(args.output_dir, "logs")
    im_dir = os.path.join(args.output_dir, "im")

    frames = [name for name in sorted(os.listdir(data_dir_ldr)) if os.path.isfile(os.path.join(data_dir_ldr, name))]

    random.seed('111')
    random.shuffle(frames)
    frames_test = frames
    testing_samples = len(frames_test)

    print("\n\nData to be used:")
    print("\t%d testing HDRs" % testing_samples)

    test_ldr_paths = []
    test_hdr_paths = []
    test_exposure = []
    test_label = []
#    brackets = list(range(50,1050,50))
#    for filename in frames_test:
#        filename = filename.strip()
#        namelist = filename.split('_')
#        for bb in brackets:
#            test_hdr_paths.append(f'{data_dir_hdr}/{filename}')
#            ldr_name = f'{filename[:-4]}_{bb}.png'
#            test_ldr_paths.append(f'{data_dir_ldr}/{ldr_name}')
#            test_exposure.append(bb)
#            test_label.append(float(namelist[3][:-4]))
    for filename in frames_test:
        filename = filename.strip()
        namelist = filename.split('_')
        ldr_name = filename
        test_ldr_paths.append(f'{data_dir_ldr}/{ldr_name}')
        test_hdr_paths.append(f'{data_dir_ldr}/{ldr_name}')
        bb = namelist[6]
        test_exposure.append(bb)
        test_label.append(float(namelist[5]))

    test_dataset = LDR2HDR2Illuminance(test_ldr_paths, test_hdr_paths, test_exposure, test_label, args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)



    dphi = 2 * np.pi/args.width
    dtheta = np.pi/args.height

    phi = torch.zeros((args.height//2, args.width))
    theta = torch.zeros((args.height//2, args.width))

    for y in range(args.height//2):
        for x in range(args.width):
            phi[y,x] = (x / args.width) * 2 * np.pi
            theta[y,x] = (y / args.height) * np.pi
    phi = phi.unsqueeze(0).repeat(args.batch_size,1,1).cuda()
    theta = theta.unsqueeze(0).repeat(args.batch_size,1,1).cuda()

    criterion_hdr = nn.MSELoss().cuda()
    criterion_reg = nn.MSELoss().cuda()


    test_loss = 0
    test_loss_reg = 0
    test_loss_hdr = 0
    test_count = 0
    test_correct_025 = 0
    test_correct_010 = 0
    for _, batch_data in enumerate(test_dataloader):
        net.eval()
        with torch.no_grad():
            ldr_test = batch_data['ldr'].cuda()
            hdr_test = batch_data['hdr'].cuda()
            label_test = batch_data['label'].cuda() / 1000.0
            exposure_test = torch.LongTensor(batch_data['exposure']).reshape(ldr_test.shape[0],1)
            exposure_test = torch.zeros(args.batch_size,20).scatter_(1,exposure_test/50-1,1)
            exposure_test = exposure_test.cuda()

            hdr_pred_test, label_pred_test = net(ldr_test, exposure_test)
#            img = 179 * (hdr_pred_test[:,0,:,:] * 0.2126 + hdr_pred_test[:,1,:,:] * 0.7152 + hdr_pred_test[:,2,:,:] * 0.0722)
#            output = img[:,0:args.height//2,:].mul(torch.sin(theta)).mul(torch.cos(theta)) * dphi * dtheta
#
#            label_pred_test = output.sum([1,2]) / 1000.0
#            for i in range(16):
#                cv.imwrite(f'test{i}.hdr', hdr_pred_test[i,:,:,:].cpu().numpy().transpose(1,2,0)[:,:,::-1])
#                cv.imwrite(f'gt{i}.hdr', hdr_test[i,:,:,:].cpu().numpy().transpose(1,2,0)[:,:,::-1])
            label_pred_test = label_pred_test.squeeze()
            label_pred_test = label_pred_test * 1.1964

#            loss_hdr_test = criterion_hdr(torch.log(hdr_pred_test+eps), torch.log(hdr_test+eps))
#            test_loss_hdr += loss_hdr_test.item()
            loss_reg_test = criterion_reg(label_pred_test, label_test.float())
            test_loss_reg += loss_reg_test.item()
#            loss_test = loss_hdr_test + loss_reg_test
#            test_loss += loss_test.item()
            test_count += label_test.shape[0]
#            print(label_pred_test)
#            print(label_test)
            test_correct_025 += ( (abs(label_pred_test-label_test) / label_test) <= 0.25 ).sum().item()
            test_correct_010 += ( (abs(label_pred_test-label_test) / label_test) <= 0.10 ).sum().item()

            print('25% Accuracy', ( (abs(label_pred_test-label_test) / label_test) <= 0.25 ).sum().item()/label_test.shape[0])
            print('10% Accuracy', ( (abs(label_pred_test-label_test) / label_test) <= 0.10 ).sum().item()/label_test.shape[0])

    print('\ntest: total loss: {0:.2f} hdr loss: {1:.2f} reg loss: {2:.2f} 25%Accuracy {3:.4f} 10%Accuracy {4:.4f}'.format(
#    test_loss / test_count,
#    test_loss_hdr / test_count,
    test_loss_reg / test_count,
    test_correct_025 / test_count,
    test_loss_reg / test_count,
    test_correct_025 / test_count,
    test_correct_010 / test_count))

    
#####################################################################################################################
#    path = 'dataset/HDR/data/2020-8-yue/2020-08-03/afternoon_kitchen_0803'
#    batch_img = []
#    batch_exposure = []
#    bracket = list(range(50,1050,50))
#    for i in range(1,17):
#        print(f'{path}/{i}.jpg')
#        img = cv.imread(f'{path}/{i}.jpg',-1)[:,:,::-1]
##        gt = cv.imread('dataset/hdr/0706_126_0_1330.hdr',-1)[:,:,::-1]
#        img = cv.resize(img, (args.width, args.height), interpolation=cv.INTER_CUBIC)
#        img = img.transpose(2,0,1) / 255.0
#        img = torch.Tensor(img)
#        batch_img.append(img.unsqueeze(0))
##        exposure = bracket[i]
##        batch_exposure.append(exposure)
#
#    batch_img = torch.cat(batch_img, dim=0).cuda()
##    batch_exposure = torch.Tensor(batch_exposure).unsqueeze(1).cuda()
#    print(batch_img.shape)
##    print(batch_exposure.shape)
##    batch_img = torch.Tensor(batch_img).unsqueeze(0).cuda()
#
#
#    with torch.no_grad():
#        hdr_pred, label_pred = net(batch_img, None)
#
#    for i in range(16):
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
    parser.add_argument("--batch_size", default=8, help='Batch size for training')

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





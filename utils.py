import torch
import numpy as np
import cv2 as cv
import lie_learn.spaces.S2 as S2
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import random


class LDR2HDR(torch.utils.data.Dataset):

    def __init__(self, ldr_paths, hdr_paths, exposure, storeid, args):
        self.ldr_paths = ldr_paths
        self.hdr_paths = hdr_paths
        self.exposure = exposure
        self.storeid = storeid
        self.args = args
        self.size = len(ldr_paths)
        self.min_ratio = 0.6
        self.max_ratio = 1.0
    
    def extract_exposure(self, x):
        return int(x.split('_')[-2])

    def switch_exposure(self, x):
        exposure = int(x.split('_')[-2])
        exposure_pool = list(range(50,1050,50))
        exposure_pool.remove(exposure)
        new_exposure = np.random.choice(exposure_pool)
        return x[:len(x)-x[::-1].find('_', 9)]+str(new_exposure)+'_8084.jpg'
    
    def __getitem__(self, index):

        ldr_paths_1, hdr_paths, exposure_1, storeid = self.ldr_paths[index], self.hdr_paths[index], self.exposure[index], self.storeid[index]
        ldr_paths_2 = self.switch_exposure(ldr_paths_1)
        exposure_2 = self.extract_exposure(ldr_paths_2)
        
        ldr_1 = cv.imread(ldr_paths_1, -1)[:,:,::-1]
        ldr_1 = ldr_1.transpose(2,0,1)
        ldr_2 = cv.imread(ldr_paths_2, -1)[:,:,::-1]
        ldr_2 = ldr_2.transpose(2,0,1)
        hdr = cv.imread(hdr_paths, -1)[:,:,::-1] * 255.0
        if storeid == '0081':
            hdr *= 1.4514
        elif storeid == '0126':
            hdr *= 1.8207
        elif storeid == '0055':
            hdr *= 1.4905
        elif storeid == '0058':
            hdr *= 1.7315
        elif storeid == '0108':
            hdr *= 1.4626
        elif storeid == '0163':
            hdr *= 1.5645
        elif storeid == '1028':
            hdr *= 1.6403
        elif storeid == '1055':
            hdr *= 1.4042
        else:
            print(storeid)
            raise ValueError("Invalid storeid!")
        hdr = hdr.transpose(2,0,1)

        ldr_1 = ldr_1 / 255.0
        ldr_2 = ldr_2 / 255.0
        hdr = hdr/ 255.0

        return {'ldr': (ldr_1, ldr_2),
                'hdr': hdr,
                'exposure': (int(exposure_1), int(exposure_2))}

    def __len__(self):
        return self.size

class PairwiseDataset(torch.utils.data.Dataset):

    def __init__(self, ldr_paths, hdr_paths, exposure, label, storeid, args):
        self.ldr_paths = ldr_paths
        self.hdr_paths = hdr_paths
        self.exposure = exposure
        self.label = label
        self.storeid = storeid
        self.args = args
        self.size = len(ldr_paths)
        
    def extract_exposure(self, x):
        return int(x.split('_')[-2])
    
    def switch_exposure(self, x):
        exposure = int(x.split('_')[-2])
        exposure_pool = list(range(50,1050,50))
        exposure_pool.remove(exposure)
        new_exposure = np.random.choice(exposure_pool)
        return x[:len(x)-x[::-1].find('_', 9)]+str(new_exposure)+'_8084.jpg'
    
    def __getitem__(self, index):

        ldr_paths_1, hdr_paths, exposure_1, label, storeid = self.ldr_paths[index], self.hdr_paths[index], self.exposure[index], self.label[index], self.storeid[index]
        ldr_paths_2 = self.switch_exposure(ldr_paths_1)
        exposure_2 = self.extract_exposure(ldr_paths_2)

        ldr_1 = cv.imread(ldr_paths_1, -1)[:,:,::-1]
#        ldr_1 = cv.resize(ldr_1, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        ldr_1 = ldr_1.transpose(2,0,1).astype(np.float32)
        ldr_1 = ldr_1 / 255.0
        ldr_2 = cv.imread(ldr_paths_2, -1)[:,:,::-1]
#        ldr_2 = cv.resize(ldr_2, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        ldr_2 = ldr_2.transpose(2,0,1).astype(np.float32)
        ldr_2 = ldr_2 / 255.0
        hdr = cv.imread(hdr_paths, -1)[:,:,::-1] * 255.0
        hdr = cv.resize(hdr, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        if storeid == '0081':
            hdr *= 1.4514
        elif storeid == '0126':
            hdr *= 1.8207
        elif storeid == '0055':
            hdr *= 1.4905
        elif storeid == '0058':
            hdr *= 1.7315
        elif storeid == '0108':
            hdr *= 1.4626
        elif storeid == '0163':
            hdr *= 1.5645
        elif storeid == '1028':
            hdr *= 1.6403
        elif storeid == '1055':
            hdr *= 1.4042
        else:
            print(storeid)
            raise ValueError("Invalid storeid!")
        hdr = hdr.transpose(2,0,1)
        hdr = hdr / 255.0
        
        return {'ldr': (ldr_1, ldr_2),
                'hdr': hdr,
                'exposure': (int(exposure_1), int(exposure_2)),
                'label': label}

    def __len__(self):
        return self.size


class HDR2Illuminance(torch.utils.data.Dataset):

    def __init__(self, hdr_paths, label, args):
        self.hdr_paths = hdr_paths
        self.label= label
        self.args = args
        self.size = len(hdr_paths)

    def __getitem__(self, index):
        hdr_paths, label = self.hdr_paths[index], self.label[index]

        hdr = cv.imread(hdr_paths, -1)[:,:,::-1]
        # hdr  = np.loadtxt(hdr_paths).reshape(self.args.height*4, self.args.width*4, 3)
        hdr = cv.resize(hdr, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        hdr = hdr.transpose(2,0,1)
        return {'hdr': hdr, 'label': label}

    def __len__(self):
        return self.size


class LDR2HDR2Illuminance(torch.utils.data.Dataset):

    def __init__(self, ldr_paths, hdr_paths, exposure, label, args):
        self.ldr_paths = ldr_paths
        self.hdr_paths = hdr_paths
        self.exposure = exposure
        self.label = label
        self.args = args
        self.size = len(ldr_paths)

    def __getitem__(self, index):
        
        ldr_paths, hdr_paths, exposure, label = self.ldr_paths[index], self.hdr_paths[index], self.exposure[index], self.label[index]

        ldr = cv.imread(ldr_paths, -1)[:,:,::-1]
        ldr = cv.resize(ldr, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        ldr = ldr.transpose(2,0,1).astype(np.float32)
        ldr = ldr / 255.0

        hdr = cv.imread(hdr_paths, -1)[:,:,::-1] * 255.0
        hdr = cv.resize(hdr, (self.args.width, self.args.height), interpolation=cv.INTER_CUBIC)
        hdr = hdr.transpose(2,0,1)
        hdr = hdr / 255.0
        return {'ldr': ldr, 'hdr': hdr, 'exposure': int(exposure), 'label': label}

    def __len__(self):
        return self.size
        
        

def tv_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.abs(img[:,:,:,:-1] - img[:,:,:,1:]))
    h_variance = torch.mean(torch.abs(img[:,:,:-1,:] - img[:,:,1:,:]))
    loss =  h_variance + w_variance
    return loss

def perceptual_loss(model, preds, targets):
    # [4,9,16] corresponds to pool1, pool2, pool3 layer of vgg16.
    targets = torch.log(1.0+10.0*targets)/torch.log(torch.Tensor([1.0+10.0]).cuda())
    targets = extract_features(model, targets, [4,9,16])
    preds = torch.log(1.0+10.0*preds)/torch.log(torch.Tensor([1.0+10.0]).cuda())
    preds = extract_features(model, preds, [4,9,16])
    
    weights = [1/len(preds)] * len(preds)
    
    loss = 0
    mse_criterion = nn.MSELoss().cuda()
    for pred, target, weight in zip(preds, targets, weights):
        loss += mse_criterion(pred, target) * weight
        
    return loss
    
def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

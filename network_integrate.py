import os
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
#from s2cnn import SO3Convolution
#from s2cnn import S2Convolution
#from s2cnn import so3_integrate
#from s2cnn import so3_near_identity_grid
#from s2cnn import s2_near_identity_grid
import numpy as np
from utils import *
from dequntization_net import Dequantization_net
from linearization_net import Linearization_net
from hallucination_net import Hallucination_net
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.HDR_net = HDR_net(args)
        
        self.args = args
        
#        self.register_buffer("integral", torch.zeros((args.batch_size,), requires_grad=False))
#        self.register_buffer("dphi", torch.tensor([2*np.pi/args.width], requires_grad=False))
#        self.register_buffer("dtheta", torch.tensor([np.pi/args.height], requires_grad=False))
        self.dphi = 2 * np.pi/args.width
        self.dtheta = np.pi/args.height
        
        phi = torch.zeros((self.args.height//2, self.args.width))
        theta = torch.zeros((self.args.height//2, self.args.width))
        
        for y in range(self.args.height//2):
            for x in range(self.args.width):
                phi[y,x] = (x / args.width) * 2 * np.pi
                theta[y,x] = (y / args.height) * np.pi
        self.register_buffer("phi", phi.unsqueeze(0).repeat(args.batch_size//torch.cuda.device_count(),1,1))
        self.register_buffer("theta", theta.unsqueeze(0).repeat(args.batch_size//torch.cuda.device_count(),1,1))
    
    def forward(self, ldr, exposure):
        _,_,hdr = self.HDR_net(ldr, exposure)
        img = 179 * (hdr[:,0,:,:] * 0.2126 + hdr[:,1,:,:] * 0.7152 + hdr[:,2,:,:] * 0.0722)
#        self.integral = torch.zeros((self.args.batch_size,), requires_grad=False).cuda()
        
        output = img[:,0:self.args.height//2,:].mul(torch.sin(self.theta)).mul(torch.cos(self.theta)) * self.dphi * self.dtheta
#        for y in range(self.args.height//2):
#            for x in range(self.args.width):
#                phi = (x / self.args.width) * 2 * np.pi
#                theta = (y / self.args.height) * np.pi
#                self.integral += img[:,y,x] * np.sin(theta) * np.cos(theta) * self.dphi * self.dtheta
        return hdr, output.sum([1,2]) / 1000.0


class Integral(nn.Module):
    def __init__(self, args):
        super(Integral, self).__init__()
        self.args = args

        self.dphi = 2 * np.pi/args.width
        self.dtheta = np.pi/args.height
        
        phi = torch.zeros((self.args.height//2, self.args.width))
        theta = torch.zeros((self.args.height//2, self.args.width))
        
        for y in range(self.args.height//2):
            for x in range(self.args.width):
                phi[y,x] = (x / args.width) * 2 * np.pi
                theta[y,x] = (y / args.height) * np.pi
        self.register_buffer("phi", phi.unsqueeze(0).repeat(args.batch_size//torch.cuda.device_count(),1,1))
        self.register_buffer("theta", theta.unsqueeze(0).repeat(args.batch_size//torch.cuda.device_count(),1,1))
    
    def forward(self, hdr):
        img = 179 * (hdr[:,0,:,:] * 0.2126 + hdr[:,1,:,:] * 0.7152 + hdr[:,2,:,:] * 0.0722)
        output = img[:,0:self.args.height//2,:].mul(torch.sin(self.theta)).mul(torch.cos(self.theta)) * self.dphi * self.dtheta
        return output.sum([1,2]) / 1000.0



class HDR_net(nn.Module):
    def __init__(self, args):
        super(HDR_net, self).__init__()

        self.dequantization = Dequantization_net()
        self.linearization = Linearization_net()
        self.halluicnation = Hallucination_net(args)

    def sample_1d(self, img, y_idx):
        b, h, c = img.shape
        b, n    = y_idx.shape
        
        b_idx = torch.arange(b).float().cuda()
        b_idx = b_idx.unsqueeze(-1)
        b_idx = b_idx.repeat(1, n)
        
        y_idx = torch.clamp(y_idx, 0, h-1)
        a_idx = torch.stack([b_idx, y_idx], axis=-1).long()
        batch_out = []
        for i in range(b):
            out = img[list(a_idx[i].T)]
            batch_out.append(out)
        output = torch.cat(batch_out, axis=0)
        return output.reshape(b,n,c)


    def interp_1d(self, img, y):
        b, h, c = img.shape
        b, n    = y.shape

        y_0 = torch.floor(y)
        y_1 = y_0 + 1

        y_0_val = self.sample_1d(img, y_0)
        y_1_val = self.sample_1d(img, y_1)

        w_0 = (y_1 - y).unsqueeze(-1)
        w_1 = (y - y_0).unsqueeze(-1)
        return w_0 * y_0_val + w_1 * y_1_val

    def apply_rf(self, x, rf):
        input_shape = x.shape
        b, k = rf.shape
        x = self.interp_1d(rf.unsqueeze(-1), (k-1)*x.reshape(b,-1))
        return x.reshape(input_shape)
        
    def forward(self, input, exposure):
        C_pred = self.dequantization(input)
        C_pred = torch.clamp(C_pred, 0, 1)
        # C_pred: output of dequantiation

        pred_invcrf = self.linearization(C_pred, exposure)
        B_pred = self.apply_rf(C_pred, pred_invcrf)

        thr = 0.12
        alpha, _ = torch.max(B_pred, 1)
        alpha = torch.clamp(torch.clamp(alpha-1.0+thr, max=0.0) / thr, min=1.0)
        alpha = alpha.unsqueeze(1).repeat(1,3,1,1)
        y_predict, vgg_conv_layers = self.halluicnation(B_pred, exposure)
        y_predict = F.relu(y_predict)
        A_pred = B_pred + alpha * y_predict

        return C_pred, B_pred, A_pred




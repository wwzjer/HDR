import os
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import numpy as np
from utils import *
from dequntization_net import Dequantization_net
from linearization_net import Linearization_net
from hallucination_net import Hallucination_net
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self, args, pretrain=True):
        super(Net, self).__init__()
        self.HDR_net = HDR_net(args)
        if pretrain:
            self.Reg_net = torch.load('best-model.pt')
        else:
            self.Reg_net = S2ConvNet_deep(args)

    def forward(self, ldr, exposure):
        _,_,hdr = self.HDR_net(ldr, exposure)
#        hdr = ldr
        output = self.Reg_net(hdr)
        return hdr, output


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


class S2ConvNet_deep(nn.Module):

    def __init__(self, args):
        super(S2ConvNet_deep, self).__init__()
        
        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 1,
                nfeature_out = 8,
                b_in  = args.bandwidth,
                b_out = args.bandwidth,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  =  8,
                nfeature_out = 16,
                b_in  = args.bandwidth,
                b_out = args.bandwidth//2,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 16,
                b_in  = args.bandwidth//2,
                b_out = args.bandwidth//2,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 24,
                b_in  = args.bandwidth//2,
                b_out = args.bandwidth//4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 24,
                b_in  = args.bandwidth//4,
                b_out = args.bandwidth//4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 32,
                b_in  = args.bandwidth//4,
                b_out = args.bandwidth//8,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = args.bandwidth//8,
                b_out = args.bandwidth//8,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64,out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=1)
        )
        self.args = args

        theta, phi = S2.meshgrid(b=args.bandwidth, grid_type="Driscoll-Healy")
        theta, phi = torch.from_numpy(theta), torch.from_numpy(phi)
        
        x_ = torch.sin(theta) * torch.cos(phi)
        y_ = torch.sin(theta) * torch.sin(phi)
        z_ = torch.cos(theta)
        
        self.register_buffer("grid", torch.stack([x_,y_,z_]))  # xyz coordinate
        self.register_buffer("ry", (theta / np.pi) * args.height)
        self.register_buffer("rx", (phi / (2*np.pi) ) * args.width)
    
        # discretize sample position
        self.register_buffer("ix", self.rx.int())
        self.register_buffer("iy", self.ry.int())

        # obtain four sample coordinates
        self.register_buffer("ix0", self.ix - 1)
        self.register_buffer("iy0", self.iy - 1)
        self.register_buffer("ix1", self.ix + 1)
        self.register_buffer("iy1", self.iy + 1)

        xmin = 0
        xmax = self.args.height
        ymin = 0
        ymax = self.args.width
        
        self.register_buffer("idxs_00", (xmin <= self.ix0.long()) & (self.ix0.long() < xmax) & (ymin <= self.iy0.long()) & (self.iy0.long() < ymax) )
        self.register_buffer("idxs_10", (xmin <= self.ix1.long()) & (self.ix1.long() < xmax) & (ymin <= self.iy0.long()) & (self.iy0.long() < ymax) )
        self.register_buffer("idxs_01", (xmin <= self.ix0.long()) & (self.ix0.long() < xmax) & (ymin <= self.iy1.long()) & (self.iy1.long() < ymax) )
        self.register_buffer("idxs_11", (xmin <= self.ix1.long()) & (self.ix1.long() < xmax) & (ymin <= self.iy1.long()) & (self.iy1.long() < ymax) )
        
#        signal_00 = Variable(torch.zeros((args.batch_size//torch.cuda.device_count(), self.args.bandwidth*2, self.args.bandwidth*2))).cuda()
#        signal_10 = Variable(torch.zeros((args.batch_size//torch.cuda.device_count(), self.args.bandwidth*2, self.args.bandwidth*2))).cuda()
#        signal_01 = Variable(torch.zeros((args.batch_size//torch.cuda.device_count(), self.args.bandwidth*2, self.args.bandwidth*2))).cuda()
#        signal_11 = Variable(torch.zeros((args.batch_size//torch.cuda.device_count(), self.args.bandwidth*2, self.args.bandwidth*2))).cuda()
#        self.register_buffer("signal_00", signal_00)
#        self.register_buffer("signal_10", signal_10)
#        self.register_buffer("signal_01", signal_01)
#        self.register_buffer("signal_11", signal_11)
    
    def sample_bilinear(self, signal, rx, ry):

        # sample signal at each four positions
#        self.signal_00.fill_(0.0)
#        self.signal_00[:, self.idxs_00] = signal[:, self.ix0.long()[self.idxs_00], self.iy0.long()[self.idxs_00]]
#
#        self.signal_10.fill_(0.0)
#        self.signal_10[:, self.idxs_10] = signal[:, self.ix1.long()[self.idxs_10], self.iy0.long()[self.idxs_10]]
#
#        self.signal_01.fill_(0.0)
#        self.signal_01[:, self.idxs_01] = signal[:, self.ix0.long()[self.idxs_01], self.iy1.long()[self.idxs_01]]
#
#        self.signal_11.fill_(0.0)
#        self.signal_11[:, self.idxs_11] = signal[:, self.ix1.long()[self.idxs_11], self.iy1.long()[self.idxs_11]]


        sample = Variable(torch.zeros((signal.shape[0], self.args.bandwidth*2, self.args.bandwidth*2)), requires_grad=False).cuda()
        sample[:, self.idxs_00] = signal[:, self.ix0.long()[self.idxs_00], self.iy0.long()[self.idxs_00]]
        signal_00 = sample

        sample = Variable(torch.zeros((signal.shape[0], self.args.bandwidth*2, self.args.bandwidth*2)), requires_grad=False).cuda()
        sample[:, self.idxs_10] = signal[:, self.ix1.long()[self.idxs_10], self.iy0.long()[self.idxs_10]]
        signal_10 = sample

        sample = Variable(torch.zeros((signal.shape[0], self.args.bandwidth*2, self.args.bandwidth*2)), requires_grad=False).cuda()
        sample[:, self.idxs_01] = signal[:, self.ix0.long()[self.idxs_01], self.iy1.long()[self.idxs_01]]
        signal_01 = sample

        sample = Variable(torch.zeros((signal.shape[0], self.args.bandwidth*2, self.args.bandwidth*2)), requires_grad=False).cuda()
        sample[:, self.idxs_11] = signal[:, self.ix1.long()[self.idxs_11], self.iy1.long()[self.idxs_11]]
        signal_11 = sample
        
        # linear interpolation in x-direction
        fx1 = (self.ix1-self.rx) * signal_00 + (self.rx-self.ix0) * signal_10
        fx2 = (self.ix1-self.rx) * signal_01 + (self.rx-self.ix0) * signal_11
        
        # linear interpolation in y-direction
        return (self.iy1 - self.ry) * fx1 + (self.ry - self.iy0) * fx2
    
    def projection(self, img):
        """
        img: predicted intermediate HDR
        """
        brightness =  179 * (img[:,0,:,:] * 0.265 + img[:,1,:,:] * 0.67 + img[:,2,:,:] * 0.065).squeeze(1)
        sample = self.sample_bilinear(brightness, self.rx, self.ry)
        # ensure that only north hemisphere gets projected
        sample *= (self.grid[1] >= 0)
        return sample.unsqueeze(1)
    
    
    def forward(self, x):
        x = self.projection(x).to(torch.float32)
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x


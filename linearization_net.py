import os
import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import *


# class CrfFeatureNet(nn.Module):
#     def __init__(self):
#         super(CrfFeatureNet, self).__init__()

#     def fc(self, exposure, inChannels, outChannels, filterSize):
#         fc = nn.Linear(20, filterSize**2*inChannels*outChannels).cuda()
#         out = fc(exposure)
#         return out

#     def conv(self, x, exposure, filterSize, outChannels, stride, padding, relu=True):
#         batch_out = []
#         weight = self.fc(exposure, inChannels=x.shape[1], outChannels=outChannels, filterSize=filterSize)
#         for n in range(x.shape[0]):
#             out = F.conv2d(x.narrow(0,n,1), weight[n].resize(outChannels,x.shape[1],filterSize,filterSize), stride=stride, padding=padding)
#             batch_out.append(out)
#         output = torch.cat(batch_out, dim=0)
#         if relu:
#             output = F.relu(output)
#         return output

#     def batch_normalization(self, input):
#         bn = nn.BatchNorm2d(input.shape[1], affine=True).cuda()
#         return bn(input)

#     def max_pool(self, input, filterSize, stride, padding):
#         output = F.max_pool2d(input=input, kernel_size=filterSize, stride=stride, padding=padding)
#         return output

#     def forward(self, ldr, exposure):
#         conv1 = self.conv(ldr, exposure, filterSize=7, outChannels=64, stride=2, padding=3, relu=False)
#         bn_conv1 = F.relu(self.batch_normalization(conv1))
#         pool1 = self.max_pool(bn_conv1, filterSize=3, stride=2, padding=1)
#         res2a_branch1 = self.conv(pool1, exposure, filterSize=1, outChannels=256, stride=1, padding=0, relu=False)
#         bn2a_branch1 = self.batch_normalization(res2a_branch1)

#         res2a_branch2a = self.conv(pool1, exposure, filterSize=1, outChannels=64, stride=1, padding=0, relu=False)
#         bn2a_branch2a = F.relu(self.batch_normalization(res2a_branch2a))
#         res2a_branch2b = self.conv(bn2a_branch2a, exposure, filterSize=3, outChannels=64, stride=1, padding=1, relu=False)
#         bn2a_branch2b = F.relu(self.batch_normalization(res2a_branch2b))
#         res2a_branch2c = self.conv(bn2a_branch2b, exposure, filterSize=1, outChannels=256, stride=1, padding=0, relu=False)
#         bn2a_branch2c = self.batch_normalization(res2a_branch2c)

#         res2a_relu = F.relu(bn2a_branch1 + bn2a_branch2c)
#         res2b_branch2a = self.conv(res2a_relu, exposure, filterSize=1, outChannels=64, stride=1, padding=0, relu=False)
#         bn2b_branch2a = F.relu(self.batch_normalization(res2b_branch2a))
#         res2b_branch2b = self.conv(bn2b_branch2a, exposure, filterSize=3, outChannels=64, stride=1, padding=1, relu=False)
#         bn2b_branch2b = F.relu(self.batch_normalization(res2b_branch2b))
#         res2b_branch2c = self.conv(bn2b_branch2b, exposure, filterSize=1, outChannels=256, stride=1, padding=0, relu=False)
#         bn2b_branch2c = self.batch_normalization(res2b_branch2c)

#         res2b_relu = F.relu(res2a_relu + bn2b_branch2c)
#         res2c_branch2a = self.conv(res2b_relu, exposure, filterSize=1, outChannels=64, stride=1, padding=0, relu=False)
#         bn2c_branch2a = F.relu(self.batch_normalization(res2c_branch2a))
#         res2c_branch2b = self.conv(bn2c_branch2a, exposure, filterSize=3, outChannels=64, stride=1, padding=1, relu=False)
#         bn2c_branch2b = F.relu(self.batch_normalization(res2c_branch2b))
#         res2c_branch2c = self.conv(bn2c_branch2b, exposure, filterSize=1, outChannels=256, stride=1, padding=0, relu=False)
#         bn2c_branch2c = self.batch_normalization(res2c_branch2c)

#         res2c_relu = F.relu(res2b_relu + bn2c_branch2c)
#         res3a_branch1 = self.conv(res2c_relu, exposure, filterSize=1, outChannels=512, stride=2, padding=0, relu=False)
#         bn3a_branch1 = self.batch_normalization(res3a_branch1)

#         res3a_branch2a = self.conv(res2c_relu, exposure, filterSize=1, outChannels=128, stride=2, padding=0, relu=False)
#         bn3a_branch2a = F.relu(self.batch_normalization(res3a_branch2a))
#         res3a_branch2b = self.conv(bn3a_branch2a, exposure, filterSize=3, outChannels=128, stride=1, padding=1, relu=False)
#         bn3a_branch2b = F.relu(self.batch_normalization(res3a_branch2b))
#         res3a_branch2c = self.conv(bn3a_branch2b, exposure, filterSize=1, outChannels=512, stride=1, padding=0, relu=False)
#         bn3a_branch2c = self.batch_normalization(res3a_branch2c)

#         res3a_relu = F.relu(bn3a_branch1 + bn3a_branch2c)
#         res3b_branch2a = self.conv(res3a_relu, exposure, filterSize=1, outChannels=128, stride=1, padding=0, relu=False)
#         bn3b_branch2a = F.relu(self.batch_normalization(res3b_branch2a))
#         res3b_branch2b = self.conv(bn3b_branch2a, exposure, filterSize=3, outChannels=128, stride=1, padding=1, relu=False)
#         bn3b_branch2b = F.relu(self.batch_normalization(res3b_branch2b))
#         res3b_branch2c = self.conv(bn3b_branch2b, exposure, filterSize=1, outChannels=512, stride=1, padding=0, relu=False)
#         bn3b_branch2c = self.batch_normalization(res3b_branch2c)

#         res3b_relu = F.relu(res3a_relu + bn3b_branch2c)
#         return torch.mean(res3b_relu, [2,3])

class CrfFeatureNet(nn.Module):
    def __init__(self):
        super(CrfFeatureNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(102, 64, 7, 2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 1, 1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, 1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, padding=0),
            nn.BatchNorm2d(256),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, padding=0),
            nn.BatchNorm2d(256),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, padding=0),
            nn.BatchNorm2d(512)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, padding=0),
            nn.BatchNorm2d(512)
        )

    def forward(self, ldr, exposure=None):
        pool1 = self.conv1(ldr)
        bn2a_branch1 = self.conv2(pool1)
        bn2a_branch2c = self.conv3(pool1)
        res2a_relu = F.relu(bn2a_branch1 + bn2a_branch2c)
        bn2b_branch2c = self.conv4(res2a_relu)
        res2b_relu = F.relu(res2a_relu + bn2b_branch2c)
        bn2c_branch2c = self.conv5(res2b_relu)
        res2c_relu = F.relu(res2b_relu + bn2c_branch2c)
        bn3a_branch1 = self.conv6(res2c_relu)
        bn3a_branch2c = self.conv7(res2c_relu)
        res3a_relu = F.relu(bn3a_branch1 + bn3a_branch2c)
        bn3b_branch2c = self.conv8(res3a_relu)

        res3b_relu = F.relu(res3a_relu + bn3b_branch2c)
        return torch.mean(res3b_relu, [2,3])

class AEInvcrfDecodeNet(nn.Module):

    def __init__(self, n_digit=2):
        super(AEInvcrfDecodeNet, self).__init__()

        self.n_digit = n_digit
        self.s = 1024
        self.n_p = 12

    def _parse(self, lines, tag):

        for line_idx, line in enumerate(lines):
            if line == tag:
                break

        s_idx = line_idx + 1

        r = []
        for idx in range(s_idx, s_idx + int(1024 / 4)):
            r += lines[idx].split()

        return np.float32(r)


    def parse_invemor(self):

        with open(os.path.join('invemor.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        b = self._parse(lines, 'B =')
        g0 = self._parse(lines, 'g0 =')
        hinv = np.stack([self._parse(lines, 'hinv(%d)=' % (i + 1)) for i in range(11)], axis=-1)

        return b, g0, hinv

    def invcrf_pca_w_2_invcrf(self, invcrf_pca_w):
        # invcrf_pca_w [b, 11]
        _, G0, HINV = self.parse_invemor()
        b = invcrf_pca_w.shape[0]

        invcrf_pca_w = invcrf_pca_w.unsqueeze(-1)

        G0 = torch.Tensor(G0).cuda()  # [   s   ]
        G0 = G0.reshape(1, -1, 1)

        HINV = torch.Tensor(HINV).cuda()  # [   s, 11]
        HINV = HINV.unsqueeze(0)  # [1, s, 11]
        HINV = HINV.repeat(b, 1, 1)

        invcrf = G0 + torch.matmul(HINV, invcrf_pca_w)
        invcrf = invcrf.squeeze(-1)
        return invcrf

    # [b, s]
    def forward(self, feature):
        dense = nn.Linear(feature.shape[1], self.n_p-1).cuda()
        feature = dense(feature)  # [b, n_p - 1]
        invcrf = self.invcrf_pca_w_2_invcrf(feature)
        return invcrf

class Linearization_net(nn.Module):

    def __init__(self):
        super(Linearization_net, self).__init__()
        self.crf_feature_net = CrfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet()
        self.pad = nn.ReflectionPad2d(1)

        self.sobel_filters = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        self.sobel_filters = np.asarray(self.sobel_filters)

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(self.sobel_filters[0]))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(self.sobel_filters[1]))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

    def _increase(self, rf):
        g = rf[:, 1:] - rf[:, :-1]
        # [b, 1023]

        min_g, _ = torch.min(g, -1, keepdim=True)
        # [b, 1]

        # r = tf.nn.relu(1e-6 - min_g)
        r = F.relu(-min_g)
        # [b, 1023]

        new_g = g + r
        # [b, 1023]

        new_g = new_g / torch.sum(new_g, -1, keepdim=True)
        # [b, 1023]

        new_rf = torch.cumsum(new_g, -1)
        # [b, 1023]

        m = nn.ConstantPad2d((1, 0, 0, 0), 0)
        new_rf = m(new_rf)
        # [b, 1024]

        return new_rf


    def sobel_edges(self, image):
        static_image_shape = image.shape
        padded = self.pad(image)
        grad_x_r = self.sobel_filter_horizontal(padded[:,0,:,:].unsqueeze(1))
        grad_y_r = self.sobel_filter_vertical(padded[:,0,:,:].unsqueeze(1))
        grad_x_g = self.sobel_filter_horizontal(padded[:,1,:,:].unsqueeze(1))
        grad_y_g = self.sobel_filter_vertical(padded[:,1,:,:].unsqueeze(1))
        grad_x_b = self.sobel_filter_horizontal(padded[:,2,:,:].unsqueeze(1))
        grad_y_b = self.sobel_filter_vertical(padded[:,2,:,:].unsqueeze(1))
        output = torch.cat([grad_x_r, grad_x_g, grad_x_b, grad_y_r, grad_y_g, grad_y_b], dim=1)
        return output
    
    def histogram_layer(self, img, max_bin):
        # histogram branch
        tmp_list = []
        for i in range(max_bin + 1):
            histo = F.relu(1 - torch.abs(img - i / float(max_bin)) * float(max_bin))
            tmp_list.append(histo)
        histogram_tensor = torch.cat(tmp_list, dim=1)
        return histogram_tensor
    
    def forward(self, img, exposure):
        # edge branch
        edge_1 = self.sobel_edges(img)
        
        feature = self.crf_feature_net( torch.cat([img, edge_1, 
                                                   self.histogram_layer(img, 4), 
                                                   self.histogram_layer(img, 8), 
                                                   self.histogram_layer(img, 16)], 
                                                   dim=1), exposure)
        # feature = tf.cast(feature, tf.float32)

        invcrf = self.ae_invcrf_decode_net(feature)
        # [b, 1024]

        invcrf = self._increase(invcrf)
        # [b, 1024]

        # invcrf = tf.cast(invcrf, tf.float32)
        # float32

        return invcrf
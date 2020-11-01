import os
import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn
#from s2cnn import SO3Convolution
#from s2cnn import S2Convolution
#from s2cnn import so3_integrate
#from s2cnn import so3_near_identity_grid
#from s2cnn import s2_near_identity_grid
import torch.utils.data as data_utils
import gzip
import pickle
import numpy as np
from torch.autograd import Variable
import argparse
from utils import *


# class Hallucination_net(nn.Module):
#     def __init__(self):
#         super(Hallucination_net, self).__init__()
    
#         self.relu = nn.ReLU()
#         self.norm = nn.BatchNorm2d(512, affine=True)
#         self.bn1 = nn.BatchNorm2d(512,affine=True).cuda()
#         self.bn2 = nn.BatchNorm2d(512,affine=True).cuda()
#         self.bn3 = nn.BatchNorm2d(256,affine=True).cuda()
#         self.bn4 = nn.BatchNorm2d(128,affine=True).cuda()
#         self.bn5 = nn.BatchNorm2d(64,affine=True).cuda()
#         self.bn6 = nn.BatchNorm2d(3,affine=True).cuda()

#         self.deconv1 = nn.ConvTranspose2d(512, 512, (4, 4), stride=2, padding=1).cuda()
#         self.deconv2 = nn.ConvTranspose2d(512, 512, (4,4), stride=2, padding=1).cuda()
#         self.deconv3 = nn.ConvTranspose2d(512, 256, (4,4), stride=2, padding=1).cuda()
#         self.deconv4 = nn.ConvTranspose2d(256, 128, (4,4), stride=2, padding=1).cuda()
#         self.deconv5 = nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1).cuda()

#     # Convolutional layers of the VGG16 model used as encoder network
#     def encoder(self, input, exposure):

#         VGG_MEAN = [103.939, 116.779, 123.68]

#         self.exposure = exposure

#         # Convert RGB to BGR
#         red, green, blue = input[:,0,:,:], input[:,1,:,:], input[:,2,:,:]
#         red, green, blue = red.unsqueeze(1), green.unsqueeze(1), blue.unsqueeze(1)
#         bgr = torch.cat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 1)

#         # Convolutional layers size 1
#         # in [b,3,320,320], conv [64,3,3,3], out [b,64,320.320]
#         out = self.conv(bgr, exposure, outChannels=64, filterSize=3, stride=1, padding=1)
#         # in [b,64,320,320], conv [64,64,3,3], out [b,64,320,320]
#         beforepool1 = self.conv(out, exposure, outChannels=64, filterSize=3, stride=1, padding=1)
#         # in [b,64,320,320], out [b,64,160,160]
#         out = F.max_pool2d(input=beforepool1, kernel_size=(2, 2), stride=2)

#         # Convolutional layers size 2
#         # in [b,64,160,160], conv [128,64,3,3], out [b,128,160,160]
#         out = self.conv(out, exposure, outChannels=128, filterSize=3, stride=1, padding=1)
#         # in [b,128,160,160], conv [128,128,3,3] out [b,128,160,160]
#         beforepool2 = self.conv(out, exposure, outChannels=128, filterSize=3, stride=1, padding=1)
#         # in [b,128,160,160], out [b,128,80,80]
#         out = F.max_pool2d(input=beforepool2, kernel_size=(2, 2), stride=2)
            
#         # Convolutional layers size 3
#         # in [b,128,80,80], conv [256,128,3,3] out [b,256,80,80]
#         out = self.conv(out, exposure, outChannels=256, filterSize=3, stride=1, padding=1)
#         # in [b,256,80,80], conv [256,256,3,3] out [b,256,80,80]
#         out = self.conv(out, exposure, outChannels=256, filterSize=3, stride=1, padding=1)
#         # in [b,256,80,80], conv [256,256,3,3] out [b,256,80,80]
#         beforepool3 = self.conv(out, exposure, outChannels=256, filterSize=3, stride=1, padding=1)
#         # in [b,256,80,80], out [b,256,40,40]
#         out = F.max_pool2d(input=beforepool3, kernel_size=(2, 2), stride=2)

#         # Convolutional layers size 4
#         # in [b,256,40,40], conv [512,256,3,3], out [b,512,40,40]
#         out = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,40,40], conv [512,512,3,3], out [b,512,40,40]
#         out = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,40,40], conv [512,512,3,3], out [b,512,40,40]
#         beforepool4 = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,40,40], out [b,512,20,20]
#         out = F.max_pool2d(input=beforepool4, kernel_size=(2, 2), stride=2)

#         # Convolutional layers size 5
#         # in [b,512,20,20], conv [512,512,3,3], out [b,512,20,20]
#         out = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,20,20], conv [512,512,3,3], out [b,512,20,20]
#         out = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,20,20], conv [512,512,3,3], out [b,512,20,20]
#         beforepool5 = self.conv(out, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         # in [b,512,20,20], out [b,512,10,10]

#         # beforepool1 [b,64,320,320]
#         # beforepool2 [b,128,160,160]
#         # beforepool3 [b,256,80,80]
#         # beforepool4 [b,512,40,40]
#         # beforepool5 [b,512,20,20]
#         out = F.max_pool2d(input=beforepool5, kernel_size=(2, 2), stride=2)

#         return out, (input, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5)
    
#     def fc(self, exposure, inChannels, outChannels, filterSize):
#         fc = nn.Linear(20, filterSize**2*inChannels*outChannels).cuda()
#         out = fc(exposure)
#         return out
    
#     def conv(self, x, exposure, outChannels, filterSize, stride, padding):
#         batch_out = []
#         weight = self.fc(exposure, inChannels=x.shape[1], outChannels=outChannels, filterSize=filterSize)
#         for n in range(x.shape[0]):
#             out = F.conv2d(x.narrow(0,n,1), weight[n].resize(outChannels,x.shape[1],filterSize,filterSize), stride=stride, padding=padding)
#             batch_out.append(out)
#         return torch.cat(batch_out, dim=0)
    
#     def deconv_layer(self, input, exposure, out_C):
#         scale = 2
#         in_C = input.shape[1]
#         out_C = int(out_C)
#         upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
#         out = upsample(input)
#         pad = nn.ReflectionPad2d(1)
#         out = pad(out)
#         out = self.conv(out, exposure, outChannels=out_C, filterSize=3, stride=1, padding=0)
#         norm = nn.BatchNorm2d(out_C, affine=True).cuda()
#         out = norm(out)
#         out = F.relu(out)
#         return out

#     def skip_connection_layer(self, input, skip_layer, exposure):
#         skip_layer = skip_layer / 255.0
#         filterSize = 1
#         outC = input.shape[1]
#         out = self.conv(torch.cat([input, skip_layer],dim=1), exposure, outChannels=outC, filterSize=filterSize, stride=1, padding=0)
#         return out
    
#     def decoder(self, input, exposure, skip_layers):
#         # input [b,512,10,10]
#         # skip_layers[0] [b,512,10,10]
#         # skip_layers[1] [b,64,320,320]
#         # skip_layers[2] [b,128,160,160]
#         # skip_layers[3] [b,256,80,80]
#         # skip_layers[4] [b,512,40,40]
#         # skip_layers[5] [b,512,40,40]

#         sb, sf, sx, sy = input.shape
#         alpha = 0.0

#         # Upsampling 1
#         out = self.deconv_layer(input, exposure, sf)

#         # Upsampling 2
#         out = self.skip_connection_layer(out, skip_layers[5], exposure)
#         out = self.deconv_layer(out, exposure, sf)
        
#         # Upsampling 3
#         out = self.skip_connection_layer(out, skip_layers[4], exposure)
#         out = self.deconv_layer(out, exposure, sf/2)

#         # Upsampling 4
#         out = self.skip_connection_layer(out, skip_layers[3], exposure)
#         out = self.deconv_layer(out, exposure, sf/4)


#         # Upsampling 5
#         out = self.skip_connection_layer(out, skip_layers[2], exposure)
#         out = self.deconv_layer(out, exposure, sf/8)


#         # Skip-connection at full size
#         out = self.skip_connection_layer(out, skip_layers[1], exposure)
        
#         # Final convolution
#         # in [b,64,320,320], conv [3,64,1,1], out [b,3,320,320]
#         out = self.conv(out, exposure, outChannels=3, filterSize=1, stride=1, padding=0)

#         # Final skip-connection
#         out = self.bn6(out)
#         out = self.relu(out)
#         out = self.skip_connection_layer(out, skip_layers[0], exposure)

#         return out

#     def forward(self, ldr, exposure):
#         # input: 0-255
#         # Encoder
#         ldr = ldr * 255.0
#         conv_layers, skip_layers = self.encoder(ldr, exposure)
#         # Fully convolutional layers on top of VGG16 conv layers
#         # in [b,512,10,10], conv [512,512,3,3], out [b,512,10,10]
        
        
#         out = self.conv(conv_layers, exposure, outChannels=512, filterSize=3, stride=1, padding=1)
#         out = self.relu(out)
        
#         # Decoder network
#         out = self.decoder(out, exposure, skip_layers)
#         return out, conv_layers


class Hallucination_net(nn.Module):
    def __init__(self, args):
        super(Hallucination_net, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.Conv2d(64, 64, 3, 1, padding=1)
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.Conv2d(128, 128, 3, 1, padding=1)
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.Conv2d(256, 256, 3, 1, padding=1)
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.Conv2d(512, 512, 3, 1, padding=1)
        )
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.Conv2d(512, 512, 3, 1, padding=1)
        )
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        if args.adding_exposure:
            self.linear = nn.Sequential(
                nn.Linear(20, 512 * (args.height//32) * (args.width//32))
            )
            
            self.fuse = nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.decoder_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip_full = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, padding=0),
            nn.Conv2d(64, 3, 1, 1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.final_skip = nn.Sequential(
            nn.Conv2d(6, 3, 1, 1, padding=0)
        )
        self.args = args

    def forward(self, ldr, exposure):
        ldr = ldr * 255.0
        # Convert RGB to BGR
        VGG_MEAN = [103.939, 116.779, 123.68]
        red, green, blue = ldr[:,0,:,:], ldr[:,1,:,:], ldr[:,2,:,:]
        red, green, blue = red.unsqueeze(1), green.unsqueeze(1), blue.unsqueeze(1)
        bgr = torch.cat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 1)
        
        # Encoder:
        beforepool1 = self.encoder_conv1(bgr)
        out = self.max_pool1(beforepool1)
        beforepool2 = self.encoder_conv2(out)
        out = self.max_pool2(beforepool2)
        beforepool3 = self.encoder_conv3(out)
        out = self.max_pool3(beforepool3)
        beforepool4 = self.encoder_conv4(out)
        out = self.max_pool4(beforepool4)
        beforepool5 = self.encoder_conv5(out)
        out = self.max_pool5(beforepool5)
        conv_layers = out
        skip_layers = (ldr, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5)
        
        # Fully convolutional layers on top of VGG16 conv layers
        if self.args.adding_exposure:
            embedding = self.linear(exposure).resize(self.args.batch_size//torch.cuda.device_count(), 512, self.args.height//32, self.args.width//32)
            out = torch.cat([out,embedding], dim=1)
            out = self.fuse(out)
        
        out = self.conv(out)
        
        
        # Decoder network
        out = self.decoder_conv1(out)
        out = self.decoder_conv2(torch.cat([out, skip_layers[5]/255.0],dim=1))
        out = self.decoder_conv3(torch.cat([out, skip_layers[4]/255.0],dim=1))
        out = self.decoder_conv4(torch.cat([out, skip_layers[3]/255.0],dim=1))
        out = self.decoder_conv5(torch.cat([out, skip_layers[2]/255.0],dim=1))
        out = self.skip_full(torch.cat([out, skip_layers[1]/255.0],dim=1))
        out = self.final_skip(torch.cat([out, skip_layers[0]/255.0],dim=1))
        return out, conv_layers

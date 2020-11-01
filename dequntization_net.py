import torch.nn.functional as F
import torch
import torch.nn as nn

#class Dequantization_net(nn.Module):
#
#    def __init__(self):
#        super(Dequantization_net, self).__init__()
#        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
#
#    def fc(self, exposure, inChannels, outChannels, filterSize):
#        fc = nn.Linear(20, filterSize**2*inChannels*outChannels).cuda()
#        out = fc(exposure)
#        return out
#
#    def conv(self, x, exposure, outChannels, filterSize, stride, padding):
#        batch_out = []
#        weight = self.fc(exposure, inChannels=x.shape[1], outChannels=outChannels, filterSize=filterSize)
#        for n in range(x.shape[0]):
#            out = F.conv2d(x.narrow(0,n,1), weight[n].resize(outChannels,x.shape[1],filterSize,filterSize), stride=stride, padding=padding)
#            batch_out.append(out)
#        return torch.cat(batch_out, dim=0)
#
#    def down(self, x, exposure, outChannels, filterSize):
#        out = F.avg_pool2d(x, 2)
#        out = self.conv(out, exposure, outChannels, filterSize, stride=1, padding=int((filterSize-1)/2))
#        out = F.leaky_relu(out, 0.1)
#        out = self.conv(out, exposure, outChannels, filterSize, stride=1, padding=int((filterSize-1)/2))
#        out = F.leaky_relu(out, 0.1)
#        return out
#
#    def up(self, x, exposure, outChannels, skpCn):
#        x = self.upsampling(x)
#        out = self.conv(x, exposure, outChannels, filterSize=3, stride=1, padding=1)
#        out = F.leaky_relu(out, 0.1)
#        out = self.conv(torch.cat([x,skpCn], dim=1), exposure, outChannels, filterSize=3, stride=1, padding=1)
#        out = F.leaky_relu(out, 0.1)
#        return out
#
#    def forward(self, input_images, exposure):
#        num_batch = input_images.shape[0]
#        x = self.conv(input_images, exposure, outChannels=16, filterSize=7, stride=1, padding=3)
#        x = F.leaky_relu(x ,0.1)
#        s1 = self.conv(x, exposure, outChannels=16, filterSize=7, stride=1, padding=3)
#        s1 = F.leaky_relu(s1, 0.1)
#
#        s2 = self.down(s1, exposure, 32, 5)
#        s3 = self.down(s2, exposure, 64, 3)
#        s4 = self.down(s3, exposure, 128, 3)
#        x = self.down(s4, exposure, 256, 3)
#        x = self.up(x, exposure,128, s4)
#        x = self.up(x, exposure, 64, s3)
#        x = self.up(x, exposure, 32, s2)
#        x = self.up(x, exposure, 16, s1)
#        x = self.conv(x, exposure, outChannels=3, filterSize=3, stride=1, padding=1)
#        x = F.tanh(x)
#        output = x + input_images
#        return output


class Dequantization_net(nn.Module):

    def __init__(self):
        super(Dequantization_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, padding=3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 7, 1, padding=3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 5, 1, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, padding=1),
            nn.Tanh()
        )

    def forward(self, input_images):
        x = self.conv1(input_images)
        s1 = self.conv2(x)

        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)

        x = self.up1(x)
        x = self.skip1(torch.cat([x, s4], dim=1))
        x = self.up2(x)
        x = self.skip2(torch.cat([x, s3], dim=1))
        x = self.up3(x)
        x = self.skip3(torch.cat([x, s2], dim=1))
        x = self.up4(x)
        x = self.skip4(torch.cat([x, s1], dim=1))

        x = self.conv3(x)
        output = x + input_images
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from lib.resnets import ResNet50


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
    def forward(self, x):
        x = self.conv_bn(x)
        return x


class DimReduce(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimReduce, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class BPM(nn.Module):
    def __init__(self, channel):
        super(BPM, self).__init__()
        self.conv1 = conv_upsample(channel)
        self.conv2 = conv_upsample(channel)
        self.conv3 = conv_upsample(channel)
        self.conv4 = conv_upsample(channel)
        self.conv5 = conv_upsample(channel)
        self.conv6 = conv_upsample(channel)
        self.conv7 = conv_upsample(channel)
        self.conv8 = conv_upsample(channel)
        self.conv9 = conv_upsample(channel)
        self.conv10 = conv_upsample(channel)
        self.conv11 = conv_upsample(channel)
        self.conv12 = conv_upsample(channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_s1 *
                                    self.conv7(x_s2, x_e1) *
                                    self.conv8(x_s3, x_e1) *
                                    self.conv9(x_s4, x_e1))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_s2 *
                                    self.conv10(x_s3, x_e2) *
                                    self.conv11(x_s4, x_e2))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_s3 *
                                    self.conv12(x_s4, x_e3))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class PyramidPooling(nn.Module):
    """
        Pyramid pooling module
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = BasicConv2d(in_channels, inter_channels, 1, **kwargs)
        self.out = BasicConv2d(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]

        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ppm1 = PyramidPooling(channel, channel)
        self.ppm2 = PyramidPooling(channel, channel)
        self.ppm3 = PyramidPooling(channel, channel)
        self.ppm4 = PyramidPooling(channel, channel)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x4 = self.ppm4(x4)
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x3 = self.ppm3(x3)
        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x2 = self.ppm2(x2)
        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x1 = self.ppm1(x1)
        x = self.output(x1)

        return x


class FSNet(nn.Module):
    def __init__(self, channel=32):
        super(FSNet, self).__init__()
        self.resnet = ResNet50()

        # Attention Channel
        self.atten_rgb_0 = self.channel_attention(64)
        self.atten_flow_0 = self.channel_attention(64)

        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.atten_rgb_1 = self.channel_attention(64 * 4)
        self.atten_flow_1 = self.channel_attention(64 * 4)

        self.atten_rgb_2 = self.channel_attention(128 * 4)
        self.atten_flow_2 = self.channel_attention(128 * 4)

        self.atten_rgb_3 = self.channel_attention(256 * 4)
        self.atten_flow_3 = self.channel_attention(256 * 4)

        self.atten_rgb_4 = self.channel_attention(512 * 4)
        self.atten_flow_4 = self.channel_attention(512 * 4)

        self.reduce_a1 = DimReduce(256, channel)
        self.reduce_a2 = DimReduce(512, channel)
        self.reduce_a3 = DimReduce(1024, channel)
        self.reduce_a4 = DimReduce(2048, channel)

        self.reduce_m1 = DimReduce(256, channel)
        self.reduce_m2 = DimReduce(512, channel)
        self.reduce_m3 = DimReduce(1024, channel)
        self.reduce_m4 = DimReduce(2048, channel)

        self.bpm1 = BPM(channel)
        self.bpm2 = BPM(channel)
        self.bpm3 = BPM(channel)
        self.bpm4 = BPM(channel)

        self.output_s = Decoder(channel)
        self.output_e = Decoder(channel)

        self.RGBF_conv3 = BasicConv2d(2048, 1024, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.initialize_weights()

    def forward(self, x, y):
        size = x.size()[2:]
        # ---- block_0 ----
        x = self.resnet.conv1_rgb(x)
        x = self.resnet.bn1_rgb(x)
        x = self.resnet.relu_rgb(x)
        y = self.resnet.conv1_flow(y)
        y = self.resnet.bn1_flow(y)
        y = self.resnet.relu_flow(y)

        atten_rgb = self.atten_rgb_0(x)
        atten_flow = self.atten_flow_0(y)
        m0 = x.mul(atten_flow) + y.mul(atten_rgb)  # (BS, 64, 176, 176)

        # ---- block_1 ----
        x = self.resnet.maxpool_rgb(x)
        y = self.resnet.maxpool_flow(y)
        m0 = self.resnet.maxpool(m0)

        x1 = self.resnet.layer1_rgb(x)
        y1 = self.resnet.layer1_flow(y)
        m1 = self.resnet.layer1(m0)

        atten_rgb = self.atten_rgb_1(x1)
        atten_flow = self.atten_flow_1(y1)
        m1 = m1 + x1.mul(atten_flow) + y1.mul(atten_rgb)  # (BS, 256, 88, 88)

        # ---- block_2 ----
        x2 = self.resnet.layer2_rgb(x1)
        y2 = self.resnet.layer2_flow(y1)
        m2 = self.resnet.layer2(m1)

        atten_rgb = self.atten_rgb_2(x2)
        atten_flow = self.atten_flow_2(y2)
        m2 = m2 + x2.mul(atten_flow) + y2.mul(atten_rgb)  # (bs, 512, 44, 44)

        # ---- block_3 ----
        x3 = self.resnet.layer3_rgb(x2)
        y3 = self.resnet.layer3_flow(y2)
        m3 = self.resnet.layer3(m2)

        atten_rgb = self.atten_rgb_3(x3)
        atten_flow = self.atten_flow_3(y3)
        m3 = m3 + x3.mul(atten_flow) + y3.mul(atten_rgb)

        # ---- block_4 ----
        x4 = self.resnet.layer4_rgb(x3)
        y4 = self.resnet.layer4_flow(y3)
        m4 = self.resnet.layer4(m3)

        atten_rgb = self.atten_rgb_4(x4)
        atten_flow = self.atten_flow_4(y4)
        m4 = m4 + x4.mul(atten_flow) + y4.mul(atten_rgb)

        # ---- feature abstraction ----
        x_s1 = self.reduce_a1(m1)
        x_s2 = self.reduce_a2(m2)
        x_s3 = self.reduce_a3(m3)
        x_s4 = self.reduce_a4(m4)

        x_e1 = self.reduce_m1(y1)
        x_e2 = self.reduce_m2(y2)
        x_e3 = self.reduce_m3(y3)
        x_e4 = self.reduce_m4(y4)

        # ---- four bi- refinement units ----
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.bpm1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.bpm2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.bpm3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.bpm4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        # ---- feature aggregation using naive u-net ----
        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e

    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)

        activation = nn.Sigmoid()       # todo: modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_flow' in k:
                name = k.split('_flow')[0] + k.split('_flow')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_rgb' in k:
                name = k.split('_rgb')[0] + k.split('_rgb')[1]
                v = pretrained_dict[name]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
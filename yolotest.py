import torch
import torch.nn as nn

import math


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        return x, feat


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        self.resblock_body1 = Resblock_body(64, 64)
        self.resblock_body2 = Resblock_body(128, 128)
        self.resblock_body3 = Resblock_body(256, 256)
        self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.num_features = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.resblock_body1(x)
        x, _ = self.resblock_body2(x)
        x, feat1 = self.resblock_body3(x)
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))
    return model


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi = phi
        self.backbone = darknet53_tiny(pretrained)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)], 384)


    def forward(self, x):
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 3:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        P5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5)
        P5_Upsample = self.upsample(P5)
        if 1 <= self.phi and self.phi <= 3:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        out1 = self.yolo_headP4(P4)
        return out0, out1
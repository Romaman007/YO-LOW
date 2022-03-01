import torch
from torch import nn
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Dense(nn.Module):
    def __init__(self,in_channels,num):
        super().__init__()
        self.blocks=nn.ModuleList()
        self.convin=Conv(in_channels,in_channels,3)
        self.convout=Conv((num+1)*in_channels,in_channels*2,1)
        for i in range(num):
            self.blocks.append(
                Conv(
                    in_channels+in_channels*i,
                    in_channels,
                    kernel_size=3,
                    stride=1
                )
            )

    def forward(self,x):
        x=self.convin(x)
        inp=x
        for block in self.blocks:
            out = block(inp)
            inp=torch.cat([inp,out],dim=1)
        return self.convout(inp)




class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = out_channels
        self.conv1 = Conv(in_channels, out_channels, 3)
        self.max_pooling = nn.MaxPool2d([2, 2], [2, 2])
        self.Dense = Dense(out_channels//2,1)

    def forward(self, x):
        x = self.conv1(x)
        skip = x
        x = torch.split(x, self.out // 2, dim=1)[1]
        x= self.Dense(x)
        x= torch.cat([x,skip],dim=1)
        x=self.max_pooling(x)
        feat = x
        return x,feat

class CSRDarkNet_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv(3,32,kernel_size=3,stride=2)
        self.conv2 = Conv(32,64,kernel_size=3,stride=2)
        self.blocks=nn.ModuleList()
        for i in range(3):
            self.blocks.append(
                CSPBlock(64*(2**i),64*(2**i))
            )
        self.conv3=Conv(512,512,3)

    def forward(self,x):
        f=[]
        x=self.conv1(x)
        x=self.conv2(x)
        for block in self.blocks:
            f.append(x)
            x,f[-1]=block(x)
        x=self.conv3(x)
        return f[1],x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='bicubic')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        Conv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class Yolomain(nn.Module):
    def __init__(self,anchors_mask,num_classes):
        super().__init__()
        self.backbone = CSRDarkNet_tiny()
        self.conv512 = Conv(512,512,3)
        self.up = Upsample(128,256)
        self.Head13 = yolo_head([512,len(anchors_mask[0])*(5+num_classes)],512)
        self.Head26 = yolo_head([256,len(anchors_mask[1])*(5+num_classes)],512)
        self.preUp = nn.Conv2d(512,128,1)

    def forward(self,x):
        f1,f2=self.backbone(x)

        x=self.conv512(f2)
        down = self.up(self.preUp(x))

        up=torch.cat([f1,down],dim=1)
        print(up.size())
        out1 = self.Head26(up)
        out2= self.Head13(x)
        return out1,out2

if __name__ == "__main__":
    anchors_mask = [[3, 4, 5], [1, 2, 3]]
    num_classes = 80
    x=torch.randn(2,3,416,416)
    model=Yolomain(anchors_mask,num_classes)
    y,y1=model(x)
    print(y.size())
    print(y1.size())

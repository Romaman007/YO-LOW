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
        self.convout=Conv((num+1)*in_channels,in_channels,1)
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
        self.conv1 = Conv(in_channels, out_channels, 3)
        self.max_pooling = nn.MaxPool2d([2, 2], [2, 2])
        self.Dense = Dense(out_channels//2,1)

    def forward(self, x):
        x = self.conv1(x)
        skip = torch.split(x, self.out // 2, dim=1)[1]
        x = torch.split(self.out // 2, x, dim=1)[1]
        x= self.Dense(x)
        x= torch.cat([x,skip],dim=1)
        feat=x
        x=self.max_pooling(x)
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
        return x,f[1]
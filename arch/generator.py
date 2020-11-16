import torch.nn as nn
import functools
class Resnet(nn.Module):
    def __init__(self,dim):
        super(Resnet, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim,dim,3,1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim,dim,3,1),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x+self.model(x)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3,64,7,1,0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(64,128,3,2,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        resnet = []
        for i in range(9):
            resnet+=[Resnet(256)]
        self.model = nn.Sequential(*resnet)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.net2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,3,7,1,0),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.net1(x)
        x = self.downsample(x)
        x = self.model(x)
        x = self.upsample(x)
        x = self.net2(x)
        return x




import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import tqdm
import os
import time
from torch.nn import init
from arch import generator
from arch import discriminator
import utils
import data_loader
import itertools

class cycleGAN(nn.Module):
    def __init__(self,args):
        super(cycleGAN, self).__init__()
        self.device = torch.device("cuda:"+str(args.cuda_id) if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.GAB = generator.Generator().to(self.device)
        self.GBA = generator.Generator().to(self.device)
        self.DA = discriminator.Discriminator().to(self.device)
        self.DB = discriminator.Discriminator().to(self.device)
        self.init_weights(self.GAB)
        self.init_weights(self.GBA)
        self.init_weights(self.DA)
        self.init_weights(self.DB)
        utils.print_networks([self.GAB,self.GBA,self.DA, self.DB], ['GAB', 'GBA', 'DA', 'DB'])

        self.optim_G = torch.optim.Adam(itertools.chain(self.GAB.parameters(),self.GBA.parameters()),lr=args.g_lr,betas=(args.beta1,args.beta2))
        self.optim_D = torch.optim.Adam(itertools.chain(self.DA.parameters(),self.DB.parameters()),lr=args.d_lr,betas=(args.beta1,args.beta2))
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # self.MSE = nn.BCEWithLogitsLoss()

        self.train_loader = data_loader.get_loader(args.img_path, args.mode, args.batch_size, args.num_workers,args.crop_size, args.img_size)


    def init_weights(self, net, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal(m.weight.data, 1.0, gain)
                init.constant(m.bias.data, 0.0)
        print('Network initialized with weights sampled from N(0,0.02).')
        net.apply(init_func)
    def train_G(self,args,x,y):
        x = x.to(self.device)
        y = y.to(self.device)
        valid = torch.ones(x.size(0), 1, 30, 30).to(self.device)
        fake = torch.zeros(x.size(0), 1, 30, 30).to(self.device)

        loss1 = self.MSE(self.DB(self.GAB(x)), valid)
        loss2 = self.MSE(self.DA(self.GBA(y)), valid)
        loss3 = self.L1(self.GBA(self.GAB(x)), x)
        loss4 = self.L1(self.GAB(self.GBA(y)), y)
        loss5 = self.L1(self.GAB(y), y)  #####identity loss
        loss6 = self.L1(self.GBA(x), x)  #####identity loss
        loss = loss1 + loss2 + args.lambda_rec * (loss3 + loss4) + args.lambda_rec * args.lambda_idt * (loss5 + loss6)
        self.optim_G.zero_grad()
        loss.backward()
        self.optim_G.step()
        return loss.item(), loss1.item(), loss3.item(), loss2.item(), loss4.item()
    def train_D(self,args,x,y):
        x = x.to(self.device)
        y = y.to(self.device)
        valid = torch.ones(x.size(0), 1, 30, 30).to(self.device)
        fake = torch.zeros(x.size(0), 1, 30, 30).to(self.device)
        self.optim_D.zero_grad()
        loss1 = self.MSE(self.DA(x), valid)
        loss2 = self.MSE(self.DA(self.GBA(y)), fake)
        loss3 = self.MSE(self.DB(y), valid)
        loss4 = self.MSE(self.DB(self.GAB(x)), fake)
        loss = (loss1 + loss2) * 0.5
        loss.backward()
        loss = (loss3 + loss4) * 0.5
        # loss = loss1+loss2+loss3+loss4
        loss.backward()
        self.optim_D.step()
        return loss1.item()+loss2.item()+loss3.item()+loss4.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()

    def updata_lr(self,epoch):
        for param_group in self.optim_G.param_groups:
            param_group['lr'] = -(0.0002 / 100) * (epoch - 100) + 0.0002
        for param_group in self.optim_D.param_groups:
            param_group['lr'] = -(0.0002 / 100) * (epoch - 100) + 0.0002
    def train(self,args):
        loss1_D = []
        loss2_D = []
        loss3_D = []
        loss4_D = []
        loss_D = []
        loss1_GAB = []
        loss2_GAB = []
        loss1_GBA = []
        loss2_GBA = []
        loss_G = []
        for epoch in range(args.train_epoch):
            if epoch >100:
                self.updata_lr(epoch)
            for i, (x, y) in tqdm.tqdm(enumerate(self.train_loader)):
                lossD, loss1D, loss2D, loss3D, loss4D = self.train_D(args,x,y)
                loss_D.append(lossD)
                loss1_D.append(loss1D)
                loss2_D.append(loss2D)
                loss3_D.append(loss3D)
                loss4_D.append(loss4D)
                if i%(args.n_critic) == 0:
                    lossG, loss1GAB, loss2GAB, loss1GBA, loss2GBA = self.train_G(args,x,y)
                    loss_G.append(lossG)
                    loss1_GAB.append(loss1GAB)
                    loss2_GAB.append(loss2GAB)
                    loss1_GBA.append(loss1GBA)
                    loss2_GBA.append(loss2GBA)

            print("epoch:", epoch + 1, "D_loss:", torch.mean(torch.FloatTensor(loss_D)))
            # print("epoch:", epoch + 1, "D1_loss:", torch.mean(torch.FloatTensor(loss1_D)))
            # print("epoch:", epoch + 1, "D2_loss:", torch.mean(torch.FloatTensor(loss2_D)))
            # print("epoch:", epoch + 1, "D3_loss:", torch.mean(torch.FloatTensor(loss3_D)))
            # print("epoch:", epoch + 1, "D4_loss:", torch.mean(torch.FloatTensor(loss4_D)))
            # print("epoch:", epoch + 1, "sumD_loss:", torch.mean(torch.FloatTensor(loss1_D))+torch.mean(torch.FloatTensor(loss2_D))+torch.mean(torch.FloatTensor(loss3_D))+torch.mean(torch.FloatTensor(loss4_D)))
            print("epoch:", epoch + 1, "G_loss:", torch.mean(torch.FloatTensor(loss_G)))
            if (epoch+1)%args.model_save_epoch == 0:
                utils.mkdir(args.model_save_dir)
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'DA': self.DA.state_dict(),
                                       'DB': self.DB.state_dict(),
                                       'GAB': self.GAB.state_dict(),
                                       'GBA': self.GBA.state_dict()},
                                      '%s/latest.ckpt' % (args.model_save_dir))
                # utils.save_checkpoint(self.G.state_dict(), os.path.join(args.model_save_dir, 'G_' + str(epoch + 1) + '.pkl'))
                # utils.save_checkpoint(self.D.state_dict(), os.path.join(args.model_save_dir, 'D_' + str(epoch + 1) + '.pkl'))


           






















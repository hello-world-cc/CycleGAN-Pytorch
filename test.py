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
import tensorboard
import itertools

class cycleGAN(nn.Module):
    def __init__(self,args):
        super(cycleGAN, self).__init__()
        self.device = torch.device("cuda:"+str(args.cuda_id)+"" if torch.cuda.is_available() else "cpu")
        self.GAB = generator.Generator().to(self.device)
        self.GBA = generator.Generator().to(self.device)
        self.DA = discriminator.Discriminator().to(self.device)
        self.DB = discriminator.Discriminator().to(self.device)
        utils.print_networks([self.GAB,self.GBA,self.DA, self.DB], ['GAB', 'GBA', 'DA', 'DB'])
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.model_save_dir))
            self.GAB.load_state_dict(ckpt['GAB'])
            self.GBA.load_state_dict(ckpt['GBA'])
            self.DA.load_state_dict(ckpt['DA'])
            self.DB.load_state_dict(ckpt['DB'])
        except:
            print(' [*] No checkpoint!')
        self.test_loader = data_loader.get_loader(args.img_path, args.mode, args.batch_size, args.num_workers,args.crop_size,
                                              args.img_size)


    def test(self,args):
        for i, (x, y) in enumerate(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                self.GAB.eval()
                test_images1 = self.GAB(x)
                self.GAB.train()
                self.GBA.eval()
                test_images2 = self.GBA(y)
                self.GBA.train()
            break

        size_figure_grid = 4
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(4):
            ax[k, 0].cla()
            ax[k, 0].imshow(np.transpose(x[k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[k, 1].cla()
            ax[k, 1].imshow(np.transpose(test_images1[k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[k, 2].cla()
            ax[k, 2].imshow(np.transpose(y[k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[k, 3].cla()
            ax[k, 3].imshow(np.transpose(test_images2[k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
        plt.show()



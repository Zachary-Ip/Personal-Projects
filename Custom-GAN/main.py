# imports
import torchvision as tv
import numpy as np
import os
import time
import torch
import datetime

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from sagan_models import Generator, Discriminator
#from utils import *

# Global variables

BATCH_SIZE = 32
EPOCHS = 10
def __init__(self):
     #setup hyper parameters
    self.imsize = 96
    self.g_num = 5
    self.z_dim = 128
    self.g_conv_dim = 64
    self.d_conv_dim = 64
    self.parallel = False

    self.lambda_gp = 10
    self.total_step = EPOCHS
    self.d_iters = 5
    self.batch_size = 64
    self.num_workers = 2
    self.g_lr = 0.0001
    self.d_lr = 0.0004
    self.lr_decay = 0.95
    self.beta1 = 0.0
    self.beta2 = 0.9
    self.pretrained_model = None

    self.use_tensorboard = False
    self.image_path = root
    self.log_path = 'C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/SAGAN/logs'
    self.model_save_path = config.model_save_path
    self.sample_path = config.sample_path
    self.log_step = config.log_step
    self.sample_step = config.sample_step
    self.model_save_step = config.model_save_step
    self.version = config.version
    
    self.build_model()
    
def save_images(model, vec, filename):
    images = model.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36],normalize = True, nrow=6)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255)
    image = Image.fromarray(ims)
    image.save(filename)
    
def main(self):
   

    os.makedirs("C:\\Users\\ipzac\\Documents\\Project Data\\Custom GAN\\results\\generated", exist_ok=True)
    os.makedirs("C:\\Users\\ipzac\\Documents\\Project Data\\Custom GAN\\results\\checkpoints", exist_ok=True)
    
    root = 'C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\Clean Sprites'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    self.dataset = ImageFolder(
            root=root,
            transform=transform
            )
    data_loader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )
    data_iter = iter(dataloader)
    step_per_epoch = len(self.data_loader)
    model_save_step = int(self.model_save_step * step_per_epoch)
        
    fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        
    # Start with trained model
    if self.pretrained_model:
        start = self.pretrained_model + 1
    else:
        start = 0
        
    # Start time
    start_time = time.time()
    for step in range(start, self.total_step):

        # ================== Train D ================== #
        self.D.train()
        self.G.train()

        try:
            real_images, _ = next(data_iter)
        except:
            data_iter = iter(self.data_loader)
            real_images, _ = next(data_iter)

            
        # Compute loss with real images
        # dr1, dr2, df1, df2, gf1, gf2 are attention scores
        real_images = tensor2var(real_images)
        d_out_real,dr1,dr2 = self.D(real_images)
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        
        
        # apply Gumbel Softmax
        z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
        fake_images,gf1,gf2 = self.G(z)
        d_out_fake,df1,df2 = self.D(fake_images)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        
        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # ================== Train G and gumbel ================== #
        # Create random noise
        z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
        fake_images,_,_ = self.G(z)
        
        # Compute loss with fake images
        g_out_fake,_,_ = self.D(fake_images)  # batch x n
        g_loss_fake = - g_out_fake.mean()
        
        
        self.reset_grad()
        g_loss_fake.backward()
        self.g_optimizer.step()


        # Print out log info
        if (step + 1) % self.log_step == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                  " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                  format(elapsed, step + 1, self.total_step, (step + 1),
                         self.total_step , d_loss_real.data[0],
                         self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))

        # Sample images
        if (step + 1) % self.sample_step == 0:
            fake_images,_,_= self.G(fixed_z)
            save_image(denorm(fake_images.data),
                       os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

        if (step+1) % model_save_step==0:
            torch.save(self.G.state_dict(),
                       os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
            torch.save(self.D.state_dict(),
                       os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
        
def build_model(self):
    
    self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        
    self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
    if self.parallel:
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)    
        
    # Loss and optimizer     
    self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])     
        
    self.c_loss = torch.nn.CrossEntropyLoss()
    # print networks
    print(self.G)
    print(self.D)    

def reset_grad(self):
    self.d_optimizer.zero_grad()
    self.g_optimizer.zero_grad()
        
def save_sample(self, data_iter):
    real_images, _ = next(data_iter)
    save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))       
        
if __name__ == "__main__":
    self  = __init__(self)
    __spec__ = None
    main(self)        
        
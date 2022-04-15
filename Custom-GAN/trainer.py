import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *

class Trainer(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
        self.model = 'sagan' # sagan or qgan
        self.adv_loss = 'wgan-gp'  #wgan-gp or hinge
        self.imsize = self.data_loader.imsize # 96
        self.g_num = 5
        self.z_dim = 128
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.parallel = True # default false
        
        self.lambda_gp = 10
        self.total_steps = 10
        self.d_iters = 5
        self. batch_size = 64
        self.num_workers = self.data_loader.num_workers # 2 
        self.g_lr = 0.0001
        self.d_lr = 0.0004
        self.lr_decay = 0.095
        self.beta1 = 0.0
        self.beta2 = 00.9
        self.pretrained_model = None
        
        self.dataset = 'pkmn'
        self.use_tensorboard = False
        self.image_path = self.data_loader.path
        self.log_path = 'C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/SAGAN/logs'
        self.model_save_path = 'C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/SAGAN/models'
        self.sample_path = 'C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/SAGAN/samples'
        self.model_save_step = 1.0
        
        self.build_model()
        
        if self.use_tensorboard:
            self.build_tensorboard()
            
        if self.pretrained_model:
            self.load_pretrained_model()
        
    def train(self):
        
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)
        
        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        
        # start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
        
        # Start time
        start_time = time.time()
        
        for step in range(start, self.total_step):
            
            # Train D
            self.D.train()
            self.G.train()
            
            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)
            
            # Compute Loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_fake, df1, df2 = self.D(fake_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif  self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf1, gf2 = self.G(z)
            d_out_fake, df1, df2 = self.D(fake_images)
            
            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0  + d_out_fake).mean()
            
            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            if self.adv_loss == 'wgan-gp':
                #compute gradient penalty
                alpha = torch.rand(real_images.size(0),1,1,1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_iamges.data + (1 - alpha) * fake_images.data, requires_grad = True)
                out,_,_ = self.D(interpolated)
                
                grad = torch.autograd.grad(outputs = out,
                                           inputs = interpolated,
                                           grad_outputs = torch.ones(out.size()).cuda(),
                                           retain_graph = True,
                                           create_graph = True,
                                           only_inputs = True)[0]
                
                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad **2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                
                # Backward + Optimize
                d_loss = self.lambda_gp *d_loss_gp
                
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
            # Train G and gumbel
            # Create noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_,_ = self.G(z)
                                        
            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_iamges) # batch x n
            if self.adv_loss =='wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()
            
            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()
        
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds = elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.data[0],
                             self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))
            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_ = self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path,'{}_fake.png'.format(step + 1)))
            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
            
    def build_model(self):
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.Batch_size, self.imsize, self.d_conv_dim).cuda()
        
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D= nn.DataParallel(self.D)
        
        # Loss and Optimizer
        
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,  [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        
        self.c_loss = torch.nn.CrossEntropyLoss()
        
        # Print networks
        print(self.G)
        print(self.D)
        
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)
        
    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

                
        
        
        
        
        
        
        
        
        
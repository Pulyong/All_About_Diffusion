import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .diffusion_utils import *

class GaussianDiffusion():
    def __init__(
        self,
        *,
        beta_schedule,
        timesteps
    ):
        self.timesteps = int(timesteps)
        self.betas = beta_scheduler(beta_schedule, self.timesteps)
        
        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        
        # for q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # for q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
        self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t*noise
    
    def get_noisy_image(self, x_start, t):
        x_noisy = self.q_sample(x_start,t=t)
        return x_noisy
    
    @torch.no_grad()
    def p_sample(self,model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        ## DDPM Eq 11
        model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )   
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def p_sample_loop(self,model, shape):
        device = next(model.parameters()).device
        
        b = shape[0]
        # start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total = self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long),i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape = (batch_size, image_size, image_size))

    def p_losses(self,denoise_model, x_start, t, noise=None, loss_type='l1'):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start,t=t,noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss
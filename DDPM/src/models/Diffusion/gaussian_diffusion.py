import math

import numpy as np
import torch

class GaussianDiffusion():
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        
        self.num_timesteps = int(betas.shape[0])
        
        # get alphas using betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # alpha bar
        self.alphas_cumprod_prev = np.append(1.0,self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        # cal for diffusion q(x_t|x_{t-1}) and otehrs
        
        # q(x_t|x_{t-1}) := N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * np.eye) = N(x_t; sqrt(alpha_t / alpha_{t-1})*x_{t-1}, (1 - alpha_t / alpha_{t-1}) * np.eye)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # betas = 1.0 - alphas
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod) # log(betas) = log(1.0 - alphas)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # q(x_{t-1}|x_t,x_0) = DDPM Eq 6,7
        # \tilde{beta_t} := 1 - \bar{\alpha_{t-1}} / (1 - \bar{alpha_t}) * beta_t
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) 
        )
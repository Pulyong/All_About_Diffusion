import math
import numpy as np
import torch

def get_names_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":

        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        return np.linspace(
            beta_start,beta_end,num_diffusion_timesteps,dtype=np.float64
        )
    elif schedule_name == "cosine":

        steps = num_diffusion_timesteps + 1
        t = torch.linspace(0, num_diffusion_timesteps, steps, dtype = torch.float64) / num_diffusion_timesteps
        alpha_bar = torch.cos((t+0.008) / (1+0.008) * math.pi * 0.5)**2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:]/alpha_bar[:-1])

        return torch.clip(betas,0,0.999)
    elif schedule_name == "sigmoid":

        steps = num_diffusion_timesteps + 1
        start = -3
        end = 3
        tau=1
        t = torch.linspace(0, num_diffusion_timesteps, steps, dtype = torch.float64) / num_diffusion_timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

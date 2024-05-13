import math
import numpy as np
import torch
import enum

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

class ModelMeanType(enum.Enum):
    '''
    Which type of output the model predicts
    '''
    
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()
    
class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()
    
class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
    
if __name__ == '__main__':
    print(repr(ModelMeanType.START_X))
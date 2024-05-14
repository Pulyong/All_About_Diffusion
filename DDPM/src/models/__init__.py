from Unet import *
from Diffusion import *
from omegaconf import OmegaConf

def build_diffusion_unet(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    diffusion_cfg = cfg['diffusion']
    unet_cfg = cfg['unet']
    
    diffusion = GaussianDiffusion(**diffusion_cfg)
    unet = Unet(**unet_cfg)
    
    return diffusion, unet
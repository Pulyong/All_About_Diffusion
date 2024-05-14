import torch
import wandb
import hydra
import omegaconf
from hydra import compose, initialize

initialize(version_base='1.3', config_path='./configs')
cfg = compose(config_name='unet_diffusion.yaml')
def eval_resolver(s: str):
    return eval(s)
omegaconf.OmegaConf.register_new_resolver("eval",eval_resolver)

print(cfg)
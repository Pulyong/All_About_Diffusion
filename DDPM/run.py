import argparse
from dotmap import DotMap
import torch
import wandb
import hydra
import omegaconf
from hydra import compose, initialize

from src.common.train_utils import set_global_seeds
from src.dataset import *
from src.common.logger import WandbTrainerLogger
from src.models import *
from src.trainers import *

def run(args):
    args = DotMap(args)
    
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    
    # Hydra
    initialize(version_base='1.3', config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    def eval_resolver(s: str):
        return eval(s)
    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    

    set_global_seeds(cfg.seed)
    device = torch.device(cfg.device)
    
    train_loader = build_dataloader(cfg.dataset)
    logger = WandbTrainerLogger(cfg)

    diffusion,unet = build_diffusion_unet(cfg.model)
    
    trainer = build_trainer(cfg=cfg.trainer,device=device,train_loader=train_loader,logger=logger,diffusion=diffusion,unet=unet)

    trainer.train()
    
    wandb.finish()
    torch.save(unet.state_dict(), f"./data/weights/{args.exp_name}.pth")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs')
    parser.add_argument('--config_name', type=str, default='unet_diffusion')
    parser.add_argument('--overrides', action='append', default=[])
    args = parser.parse_args()
    
    run(vars(args))
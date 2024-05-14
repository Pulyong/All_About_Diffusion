from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .imagenet1k import load_imagenet1k

def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    train_dataloadeer_cfg = cfg['train_dataloader']
    test_dataloadeer_cfg = cfg['test_dataloader']
    
    train_dataset = load_imagenet1k(is_train=True)
    #test_dataset = load_imagenet1k(is_train=False)
    
    train_loader = DataLoader(train_dataset, **train_dataloadeer_cfg, shuffle=True)
    #test_loader = DataLoader(test_dataset,**test_dataloadeer_cfg, shuffle=False)
    
    return train_loader #, test_loader
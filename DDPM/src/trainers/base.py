import numpy as np
import tqdm
import random
import wandb
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.common.schedulers import CosineAnnealingWarmUpRestarts

class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 logger,
                 diffusion,
                 unet):
        super().__init__()
        
        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.logger = logger
        
        self.diffusion = diffusion
        self.unet = unet.to(self.device)
        
        self.optimizer = self._build_optimizer(cfg.optimizer_type, cfg.optimizer)
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.scheduler)
        
        self.epoch = 0
        self.step = 0
        
    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.unet.parameters(), **optimizer_cfg)
        elif optimizer_type == 'adam':
            return optim.Adam(self.unet.parameters(), **optimizer_cfg)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.unet.parameters(), **optimizer_cfg)
        else:
            raise NotImplementedError
        
    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmUpRestarts(optimizer=optimizer, **scheduler_cfg)
    
    def train(self):
        cfg = self.cfg
        num_epochs = cfg.epochs
        loss_type = cfg.loss
        
        self.logger.log_to_wandb(self.step)
        
        for _ in range(int(num_epochs)):
            #train
            self.unet.train()
            for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
                batch_size = batch['pixel_values'].shape[0]
                batch = batch['pixel_values'].to(self.device)
                train_logs = {}
                
                # sample t uniformally
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,),device=self.device).long()
                
                loss = self.diffusion.p_losses(self.unet, batch, t, loss_type=loss_type)
                
                train_logs['train_loss'] = loss.item()
                train_logs['lr'] = self.lr_scheduler.get_lr()[0]
                
                self.logger.update_log(**train_logs)
                
                if self.step % cfg.log_every == 0:
                    self.logger.log_to_wandb(self.step)
                
                loss.backward()
                self.optimizer.step()
                
                
                self.step += 1
                self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.epoch += 1
                
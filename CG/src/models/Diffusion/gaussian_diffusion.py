import math

import numpy as np
import torch

from diffusion_utils import *

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
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod) # log(var) = log(1.0 - alpha bar)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # q(x_{t-1}|x_t,x_0) = DDPM Eq 6,7
        # \tilde{beta_t} := 1 - \bar{\alpha_{t-1}} / (1 - \bar{alpha_t}) * beta_t
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) 
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        
        # DDPM Eq 7, mean function
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
    def q_mean_variance(self, x_start, t):
        '''
        get distribution parameters of  q(x_t | x_0)
        x_start : noiseless input
        t: step
        '''
        #sqrt(alpha bar) * x_0
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod,t,x_start.shape) * x_start
        ) 
        variance = _extract_into_tensor(1.0 - self.sqrt_alphas_cumprod, t, x_start.shape) # 1 - alpha bar
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape) # log(1 - alpha bar)
        
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise = None):
        '''
        diffusion forward process
        q(x_t | x_0)
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
            
        assert noise.shape == x_start.shape
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        
    def q_posterior_mean_variance(self, x_start, x_t, t):
        '''
        compute mean & variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        DDPM Eq 7
        '''
        
        assert x_start.shape == x_t.shape
        
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        '''
        p(x_{t-1} | x_t) & prediction x_0
        '''
        if model_kwargs is None:
            model_kwargs = {}
            
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C*2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim = 1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(
                    np.log(self.betas, t, x.shape)
                )
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else: # not learned Variance
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE:(
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:]))
                ),
                ModelVarType.FIXED_SMALL:(
                    self.posterioir_variance,
                    self.posterior_log_variance_clipped
                )
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1,1)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
            
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else: # model predict epsilon
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )    
        else:
            raise NotImplementedError(self.model_mean_type)
        
        assert(
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return{
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        # from DDIM Eq 4
        assert x_t.shape == eps.shape
        return(
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        # (xprev - coef2*x_t) / coef1
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )
        
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return(
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    # DDPM Eq 7
    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised = clip_denoised,
            denoised_fn = denoised_fn, model_kwargs = model_kwargs,
            device = device, progress = progress
        ):
            final = sample
        return final['sample']
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm.auto import tqdm
            
            indices = tqdm(indices)
            
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
        

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    '''
    extract arr[timestep] and matching shape
    '''
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)        
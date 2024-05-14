import torch

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def beta_schedule(schedule, timesteps):
    if schedule == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif schedule == 'quadratic':
        betas = quadratic_beta_schedule(timesteps)
    elif schedule == 'sigmoid':
        betas = sigmoid_beta_schedule(timesteps)
    elif schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise NotImplementedError(f"{schedule} is Not Implemented")
    return betas

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape) - 1))).to(t.device)

if __name__ == '__main__':
    arr = torch.range(0,10,1)
    t = torch.range(0,3,1).type(torch.int64)
    a = extract(arr, t, (3,256,256))
    print(a.size())
    
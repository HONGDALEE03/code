import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
import math
import numpy as np
from einops import rearrange

# 辅助模块 --------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out*2)) if time_emb_dim else None
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(16, dim),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(16, dim_out),
            nn.Mish(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        self.gnorm1 = nn.GroupNorm(16, dim_out)
        self.gnorm2 = nn.GroupNorm(16, dim_out)
        self.mish = nn.Mish()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t):
        scale_shift = None
        if self.mlp and t is not None:
            t = self.mlp(t)
            t = rearrange(t, 'b c -> b c 1 1')
            scale_shift = t.chunk(2, dim=1)
        
        h = self.gnorm1(self.mish(self.block1(x)))
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        h = self.gnorm2(self.mish(self.block2(h)))
        return h + self.res_conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# 改进的ResUNet结构 --------------------------------------------------
class ResUNet(nn.Module):
    def __init__(
        self,
        dim,
        time_emb_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=16,
        output_dim=3
        
    ):
        super().__init__()
        # 初始化卷积
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 下采样模块
        self.downs = nn.ModuleList()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            self.downs.append(ResnetBlock(in_dim, out_dim, time_emb_dim=time_emb_dim))
            
        # 上采样模块
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            in_dim = dims[i + 1]
            out_dim = dims[i]
            self.ups.append(ResnetBlock(in_dim * 2, out_dim, time_emb_dim=time_emb_dim))
        
        # 最终输出
        self.final_conv = nn.Conv2d(dim, output_dim, 3, padding=1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)
        
        # 下采样
        skips = []
        for block in self.downs:
            x = block(x, t)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        
        # 上采样
        for block in self.ups:
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t)
            
        return self.final_conv(x)
    
# 双ResUNet网络定义 --------------------------------------------------
class DualUNet(nn.Module):
    def __init__(
        self,
        dim=128,
        time_emb_dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=16
        
    ):
        super().__init__()
        # 独立残差预测网络
        self.res_UNet = ResUNet(
            dim=dim,
            time_emb_dim=time_emb_dim,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
            output_dim=channels  # 预测残差
        )
        
        # 独立噪声预测网络 
        self.noise_UNet = ResUNet(
            dim=dim,
            time_emb_dim=time_emb_dim,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
            output_dim=channels  # 预测噪声
        )

    def forward(self, x, time):
        # 独立前向传播
        res_pred = self.res_UNet(x, time)
        noise_pred = self.noise_UNet(x, time)
        return res_pred, noise_pred

# 改进的扩散过程 --------------------------------------------------
class RDDM(nn.Module):
    def __init__(
        self,
        model,
        timesteps,
        alpha_schedule='cosine_decay',  # 残差系数调度
        beta_schedule='cosine_increase',
        posterior_variance = 'posterior_variance' # 噪声系数调度
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # 系数调度生成
        alpha_cumsum = self._create_schedule(alpha_schedule)
        beta_cumsum = self._create_schedule(beta_schedule)
        posterior_variance = self._create_schedule(posterior_variance)
        
        # 注册缓冲区
        self.register_buffer('alpha_cumsum', alpha_cumsum)
        self.register_buffer('beta_cumsum', beta_cumsum)
        self.register_buffer('posterior_variance', posterior_variance)

    def _create_schedule(self, type):
        """创建系数调度表"""
        s = 0.008
        steps = torch.arange(0, self.timesteps+1, dtype=torch.float32)
        alpha = torch.cos((steps / self.timesteps) + s) / (1 + s) * math.pi * 0.5
        vals = torch.clip(alpha[1:] / alpha[:-1], 0, 0.999)
        vals = torch.cat((torch.tensor([1.]), vals))
        vals = torch.cumprod(vals, dim=0)

        if type == 'cosine_decay':
            vals = 1. - torch.sqrt(vals) 
        elif type == 'cosine_increase':
            vals = torch.sqrt(1. - vals) 
        elif type == 'posterior_variance':
            vals_prev = F.pad(vals[:-1], (1, 0), value=1.)
            vals = vals_prev*(1. - vals/vals_prev)*(1. - vals_prev)/(1. - vals)
            vals[0] = torch.tensor(0.)
        return vals
    
    @torch.no_grad()
    def forward_diffusion_sample(self, x_start, t):
        """前向扩散过程（公式7）"""
        noise = torch.randn_like(x_start)
        degraded_img = torch.zeros_like(x_start)
        res = degraded_img - x_start

        alpha_t = self.alpha_cumsum[t]
        beta_t = self.beta_cumsum[t]
        x_t = x_start + alpha_t * res + beta_t * noise
        return x_t, res, noise

    @torch.no_grad()
    def timestep_sample(self, x_t, t):
        """反向采样步骤（公式13）"""
        res_pred, noise_pred = self.model(x_t, t)
        
        alpha_t = self.alpha_cumsum[t]
        alpha_prev = self.alpha_cumsum[t-1] if t>0 else 0
        beta_t = self.beta_cumsum[t]
        beta_prev = self.beta_cumsum[t-1] if t>0 else 0
        variance_t = self.posterior_variance[t]
        
        noise = torch.randn_like(x_t)
        x_prev = x_t - (alpha_t - alpha_prev)*res_pred - (beta_t - torch.sqrt(beta_prev**2 - variance_t))*noise_pred + torch.sqrt(variance_t)*noise
        return x_prev
    
    @staticmethod
    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :] 
        plt.imshow(reverse_transforms(image))
        plt.xticks([])
        plt.yticks([])
        pass
    
    @torch.no_grad()
    def sample(self, x_0, num_samples):
        """Generate samples from the model""" 
        noisy_images = []
        flag = [self.timesteps]
        stepsize = (self.timesteps-1) // num_samples

        t = torch.full((x_0.shape[0],), self.timesteps, dtype=torch.long)
        noisy_image, _, _ = self.forward_diffusion_sample(x_0, t)
        noisy_images.append(noisy_image.detach().cpu())

        for i in reversed(range(1,self.timesteps+1)):
            t = torch.full((x_0.shape[0],), i, dtype=torch.long)
            noisy_image = self.timestep_sample(noisy_image, t)
            if (i-1) % stepsize == 0:
                flag.append(i-1)
                noisy_images.append(noisy_image.detach().cpu())
        return noisy_images, flag

if __name__ == '__main__':
    data = torch.zeros([1, 3, 256, 256])
    T = 30
    model = DualUNet()
    diffusion = RDDM(model, T)
    print(diffusion.alpha_cumsum)
    print(diffusion.beta_cumsum)
    print(diffusion.posterior_variance)
    
    t = torch.randint(0, T, (data.shape[0],), )
    res, noise = model(data, t)
    print(res.shape, noise.shape)
    
    # Test forward diffusion
    noisy_image, res, noise = diffusion.forward_diffusion_sample(data, t)
    print(noisy_image.shape, res.shape, noise.shape)
    
    # Test sampling
    samples, _ = diffusion.sample(data, 5)
    print(len(samples), samples[0].shape)
import torch
import torch.nn as nn
import torch.nn.functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock, VAE_Decoder

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (N, C, H, W) -> (N, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (N, 128, H, W) -> (N, 128, H, W) -> (N, 128, H, W)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (N, 128, H, W) -> (N, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (N, 128, H/2, W/2) -> (N, 256, H/2, W/2) -> (N, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (N, 256, H/2, W/2) -> (N, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (N, 256, H/4, W/4) -> (N, 512, H/4, W/4) -> (N, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # (N, 512, H/4, W/4) -> (N, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (N, 512, H/8, W/8) -> (N, 512, H/8, W/8) -> (N, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512), 

            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # (N, 512, H/8, W/8) -> (N, 8, H/8, W/8) -> (N, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: (N, Channel, Hight, Wight))
        noise: (N, out_channels, H/8, W/8)
        """
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (batch_size, 8, H / 8, W / 8) -> two tensors of shape (batch_size, 4, H / 8, W / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance) 
        # X = mean + stdev * Z
        x = mean + stdev * noise
        # Scale the output by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        return x


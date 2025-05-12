import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)

class AE_model(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        # 初始编码层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # 16个残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(16)]
        )
        
        # 最终编码层（带残差连接）
        self.final_encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        self.flatten =  nn.Sequential(
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Linear(32 * 64 * 64, 2*base_channels)
        self.decoder_fc = nn.Linear(2*base_channels, 32 * 64 * 64)
        
        # 解码器
        self.decoder = nn.Sequential(
            # 第一层解码模块
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, padding=1),
            PixelShuffleUpsample(base_channels*4, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # 第二层解码模块
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, padding=1),
            PixelShuffleUpsample(base_channels*4, base_channels//2),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(),
            
            # 最终输出层
            nn.Conv2d(base_channels//2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 初始编码
        initial_out = self.initial_conv(x)
        # 残差块处理
        residual_out = self.residual_blocks(initial_out)
        # 最终编码（带残差）
        encoded = self.final_encoder(residual_out) + initial_out
        z = self.flatten(encoded)
        z = self.fc(z)
        h = self.decoder_fc(z)
        h = h.view(-1, 32, 64, 64)
        # 解码
        decoded = self.decoder(h)
        
        return decoded, z


if __name__=='__main__':
  
  data = torch.zeros([11, 3, 64, 64])
  model = AE_model()
  output, _ = model(data)
  print(output.shape)



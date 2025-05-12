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

class Generator(nn.Module):
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
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, padding=1),
            PixelShuffleUpsample(base_channels*4, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, padding=1),
            PixelShuffleUpsample(base_channels*4, base_channels//2),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(),
            
            nn.Conv2d(base_channels//2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        initial_out = self.initial_conv(x)
        residual_out = self.residual_blocks(initial_out)
        encoded = self.final_encoder(residual_out) + initial_out
        decoded = self.decoder(encoded)
        
        return decoded

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 3 x 256 x 256
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 128 x 128
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 16 x 16

            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


if __name__=='__main__':
    data = torch.zeros([11, 3, 64, 64])
    model = Generator()
    output = model(data)
    print(output.shape)
    data = torch.zeros([11, 3, 256, 256])
    model = Discriminator()
    output = model(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_fake = torch.full((11,), 0.0, device=device)  
    print(label_fake.shape)
    print(output.shape)
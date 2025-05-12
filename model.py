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
        out += residual 
        out = F.relu(out)
        return out

class VAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
   
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(8)]
        )
 
        self.final_encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        self.flatten =  nn.Sequential(
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(base_channels* 64 * 64, 4 * base_channels)
        self.fc_log_var = nn.Sequential(
            nn.Linear(base_channels * 64 * 64, 4 * base_channels),
            nn.Softplus() 
        )
          
        self.decoder_fc = nn.Linear(4 * base_channels, base_channels * 64 * 64)
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
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
     
    def forward(self, x):
        initial_out = self.initial_conv(x)
        residual_out = self.residual_blocks(initial_out)
        encoded = self.final_encoder(residual_out) + initial_out
        z = self.flatten(encoded)

        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        z = self.reparameterize(mu, log_var)
        h = self.decoder_fc(z)
        h = h.view(-1, 32, 64, 64)
    
        decoded = self.decoder(h)
        
        return decoded, mu, log_var
    
if __name__=='__main__':
    data = torch.zeros([11, 3, 64, 64])
    model = VAE()
    output, _, _ = model(data)
    print(output.shape)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from model import Generator, Discriminator

os.chdir("D:\\code\\GAN")
class StressDataset(Dataset):
    def __init__(self, stress_dir, transform=None):
        self.stress_dir = stress_dir
        self.stress_files = [os.path.join(stress_dir, f) for f in os.listdir(stress_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.stress_files)

    def __getitem__(self, idx):
        img_path = self.stress_files[idx]
        stress = Image.open(img_path).convert('RGB')
        stress1 = np.array(stress).astype(np.float32).transpose(2, 0, 1)/255.0

        if self.transform:
            stress = self.transform(stress)
        return stress, stress1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

traindataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
    transform=transform
)

dataloader = DataLoader(traindataset, batch_size=8, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
netG.load_state_dict(torch.load('gan_generator.pth'))
# netD.load_state_dict(torch.load('gan_discriminator.pth'))

def criterion(input, target):
    BCL = nn.functional.mse_loss(input, target)
    return BCL

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
schedulerG = StepLR(optimizerG, step_size=600, gamma=0.1)
schedulerD = StepLR(optimizerD, step_size=600, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    total_lossG = 0
    total_lossD = 0
    for i, (stress, stress1) in enumerate(dataloader):
        real = stress1.to(device)
        batch_size = real.size(0)
        
        label_real = torch.full((batch_size,), 1.0, device=device)
        label_fake = torch.full((batch_size,), 0.0, device=device)

        #生成器
        optimizerG.zero_grad()
        g_output = netG(stress.to(device))
        output_gen = netD(g_output)
        g_loss = 1e-3 * criterion(output_gen.view(-1), label_real)
        mse_loss = nn.functional.mse_loss(g_output, real.detach())
        
        errG = g_loss + mse_loss
        errG.backward()
        optimizerG.step()
        schedulerG.step()

        #判别器
        optimizerD.zero_grad()
        output_real = netD(real)
        errD_real = criterion(output_real.view(-1), label_real)

        output_fake = netD(g_output.detach())
        errD_fake = criterion(output_fake.view(-1), label_fake)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        schedulerD.step()

        total_lossG += errG.item()
        total_lossD += errD.item()

        if i % 10 == 0:
            print(f"batch_idx:[{i}/{len(dataloader)}] Loss_G: {errG.item():.4f} Loss_D: {errD.item():.8f} ")
            if errG.item() < 0.01:
               torch.save(netG.state_dict(), 'gan_generator.pth')
               torch.save(netD.state_dict(), 'gan_discriminator.pth')
               print('model saved!')
               break
        
    print(f"== Epoch [{epoch+1}/{num_epochs}], avrage LossG: {total_lossG / (i+1):.4f}, avrage LossD: {total_lossD / (i+1):.8f} ==")
    if total_lossG / (i+1) <= 0.0001:
        break
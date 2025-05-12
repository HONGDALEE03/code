import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset 
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
from model import VAE

os.chdir("D:\\code\\VAE")
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
        return stress,stress1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

traindataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
    transform=transform
)
dataloader = DataLoader(traindataset, batch_size=8, shuffle=None, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load('vae_model_state_dict.pth'))
model.eval()
with torch.no_grad(): 
    for i, (stress, stress1) in enumerate(dataloader):
        stress = stress[0:1].to(device)
        recon_stress, _ , _ = model(stress)
        stress1 = stress1[0:1].to(device)
        temp1 = stress1.numpy().squeeze().transpose(1,2,0)
        temp2 = recon_stress.numpy().squeeze().transpose(1,2,0)
        break

mus = []
log_vars = []
with torch.no_grad():
    for batch_idx, (stress, stress1) in enumerate(dataloader):
        stress = stress.to(device)
        _, mu, log_var = model(stress)
        mus.append(mu.cpu().numpy())
        log_vars.append(log_var.cpu().numpy())

mus = np.concatenate(mus, axis=0)
log_vars = np.concatenate(log_vars, axis=0)
vars = np.exp(0.5 * log_vars)  

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(temp1)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(temp2)
plt.title("VAE")
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(mus.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Distribution of μ (Mean)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(vars.flatten(), bins=50, color='red', alpha=0.7)
plt.title("Distribution of σ (Standard Deviation)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
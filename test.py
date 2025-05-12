import torch
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
from model import Generator
from PIL import Image
import os

os.chdir("D:\\code\\GAN")
class StressDataset(Dataset):
    def __init__(self, stress_dir, transform=None):
        self.stress_dir = stress_dir
        self.stress_files = [os.path.join(stress_dir, f) for f in os.listdir(stress_dir) if f.endswith('.png')]
        transform = transform 

    def __len__(self):
        return len(self.stress_files)

    def __getitem__(self, idx):
        img_path = self.stress_files[idx]
        stress = Image.open(img_path).convert('RGB')
        stress1 = np.array(stress).astype(np.float32).transpose(2, 0, 1)/255.0

        if transform:
            stress = transform(stress)
        # stress = np.array(stress)
        # plt.imshow(stress)
        # plt.show()
        return stress, stress1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load('gan_generator.pth'))
netG.eval()

testdataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
)
dataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

plt.close('all')

if __name__=='__main__':

    with torch.no_grad():
        for i, (stress, stress1) in enumerate(dataloader):
            stress = stress.to(device)
            recon_stress = netG(stress)
            temp1 = stress1.numpy().squeeze().transpose(1,2,0)
            temp2 = recon_stress.numpy().squeeze().transpose(1,2,0)
            plt.figure(1)
            plt.subplot(1,2,1)
            plt.imshow(temp1)
            plt.title("Original")

            plt.subplot(1,2,2)
            plt.imshow(temp2)
            plt.title("GAN")
            plt.show()
            break
    
    output_dir = "D:\code\GENERATE"
    os.makedirs(output_dir, exist_ok=True)
    vutils.save_image(recon_stress, os.path.join(output_dir, "gan_generated_stress.png"), nrow=8, normalize=True)
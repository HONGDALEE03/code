import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from PIL import Image
from model import DualUNet, RDDM
import os

os.chdir("D:\\code\\Diffusion")
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

        if transform:
            stress = transform(stress)
        return stress

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

testdataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
)
dataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

T = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = DualUNet().to(device)
model.load_state_dict(torch.load("diffusion.pth"))

diffusion = RDDM(model=model, timesteps=T)
model.eval()

plt.close('all')

if __name__=='__main__':
    num_samples = 9
    stepsize = T//num_samples
    stress = next(iter(dataloader)).to(device)
    
    plt.figure(figsize=(12,6))
    plt.axis('off')
    for idx in range(0, T+1, stepsize):
        t = torch.full((stress.shape[0],), idx, dtype=torch.long)
        plt.subplot(1, num_samples+1, idx//stepsize+1)
        noisy_img, res, noise = diffusion.forward_diffusion_sample(stress, t)
        noisy_img = torch.clamp(noisy_img, -1.0, 1.0)
        diffusion.show_tensor_image(noisy_img.detach().cpu())
        plt.title(f"t={idx}")

    plt.figure(figsize=(12,6))
    plt.axis('off')
    imgs, flag = diffusion.sample(stress, num_samples)
    for i, img in enumerate(imgs):
        img = torch.tensor(img)
        img = torch.clamp(img, -1.0, 1.0)
        plt.subplot(1, len(imgs), i+1)
        diffusion.show_tensor_image(img.detach().cpu())
        plt.title(f"t={flag[i]}")

    plt.figure(figsize=(12,6))
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(stress.numpy().squeeze().transpose(1,2,0))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(imgs[-1].numpy().squeeze().transpose(1,2,0))
    plt.title("RDDM")
    plt.show()
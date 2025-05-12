import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler 
import os
from PIL import Image
import math
from model import RDDM, DualUNet

os.chdir(r"D:\code\Diffusion")
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
        
        if self.transform:
            stress = self.transform(stress)
        return stress

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

traindataset = StressDataset(
    stress_dir=r"D:\\cosegGuitar\\256x256\\stress",
    transform=transform
)

dataloader = DataLoader(traindataset, batch_size=8, shuffle=True, num_workers=0
)

T = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualUNet().to(device)
if os.path.exists("diffusion.pth"):
    model.load_state_dict(torch.load("diffusion.pth"))

diffusion = RDDM(model=model, timesteps=T)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,       
    weight_decay=1e-2
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    pct_start=0.3,  
    steps_per_epoch=len(dataloader),
    epochs=20      
)


def get_loss(model, x_start, t):
    x_t, res, noise = diffusion.forward_diffusion_sample(x_start, t)
    res_pred, noise_pred = model(x_t, t)

    progress = t.float() / T
    res_weight = torch.cos(progress * math.pi * 0.5)  
    noise_weight = torch.sin(progress * math.pi * 0.5)  
    
    res_loss = (res_weight.view(-1,1,1,1) * F.l1_loss(res_pred, res, reduction='none')).mean()
    noise_loss = (noise_weight.view(-1,1,1,1) * F.mse_loss(noise_pred, noise, reduction='none')).mean()

    return res_loss + noise_loss

best_loss = 0.01
for epoch in range(10):
    total_loss = 0
    for i, stress in enumerate(dataloader):
        stress = stress[0:1].to(device)

        optimizer.zero_grad()
        t = torch.randint(0, T+1, (stress.shape[0],), device=device)
        loss = get_loss(model, stress, t)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  
        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            print(f"Epoches [{epoch+1}/{10}], Batchidx [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "diffusion.pth")
            if best_loss < 0.005:
                print("模型保存成功！")
                break
    print(f"== Epoch [{epoch+1}/{10}], avrage Loss: {total_loss / (i+1):.4f} ==")
    if total_loss / (i+1) < best_loss:
        break
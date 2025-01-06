from model import WaveletSuperResolutionNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import MyDataset , Resize, Crop , Augmentation

DATASET_PATH ="D:/Datasets/SuperResolution/"

# Load the model
model = WaveletSuperResolutionNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
transform = transforms.Compose([transforms.ToTensor()])

LR_path =  DATASET_PATH + "dataset/train/low_res"
GT_path =  DATASET_PATH + "dataset/train/high_res"
batch_size = 16
patch_size = 32
scale = 4

transform = transforms.Compose([
Resize(lr_size=32, gt_size=128),  # Example resizing
Crop(scale=scale, patch_size=patch_size),
Augmentation()
])


dataset = MyDataset(LR_path=LR_path, GT_path=GT_path, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):        
        LR_imgs = batch['LR']
        GT_imgs = batch['GT']
        print(LR_imgs.shape)
        optimizer.zero_grad()
        outputs = model(LR_imgs)
        print(outputs.shape)
        loss = criterion(outputs, GT_imgs)
        break
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss {loss.item()}')
    break
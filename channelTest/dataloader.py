import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils import *
from PIL import Image
from torch.utils.data import DataLoader

# Path to the dataset
DATASET_PATH ="D:/Datasets/SuperResolution/"

class INRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.high_res_paths = []
        self.low_res_paths = []
        
        # Load data from directory
        self.class_names = os.listdir(root_dir)
        
        
        self.high_res_paths += [(img, self.class_names[0]) for img in os.listdir(os.path.join(root_dir, self.class_names[0]))]
        self.low_res_paths += [(img, self.class_names[1]) for img in os.listdir(os.path.join(root_dir, self.class_names[1]))]        
        
            

    def __len__(self):
        return len(self.high_res_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img , class_name = self.high_res_paths[idx]
        high_res_path = os.path.join(self.root_dir, class_name, img)
         
        img , class_name = self.low_res_paths[idx]        
        low_res_path = os.path.join(self.root_dir, class_name, img)
        
        high_res_img = Image.open(high_res_path).convert('RGB')
        high_res_img = np.array(high_res_img)
        
        
        low_res_img = Image.open(low_res_path).convert('RGB')
        low_res_img = np.array(low_res_img)
        
        
        sample = {'high_res': high_res_img, 'low_res': low_res_img}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        

if __name__ == "__main__":
    dataset = INRDataset(DATASET_PATH + "dataset/raw_data")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # Randomly sample a batch of data
    for i, sample in enumerate(dataloader):
        high_res = sample['high_res']
        low_res = sample['low_res']
        print(f"High-res batch shape: {high_res.shape}, Low-res batch shape: {low_res.shape}")
        break
    
    high_res = high_res.permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
    low_res = low_res.permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
    
    # Display a sample high-res and low-res image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plt.imshow(high_res[0].permute(1, 2, 0))
    plt.title("High-Resolution Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(low_res[0].permute(1, 2, 0))
    plt.title("Low-Resolution Image")
    plt.axis('off')
    
    plt.show()
    

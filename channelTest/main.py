import torch
import torch.nn as nn
import pywt
import numpy as np
from dataloader import INRDataset
from model import WaveletCoefficientUpscaler
from utils import *

epochs = 10
DATASET_PATH ="D:/Datasets/SuperResolution/"



# Define your model, criterion, and optimizer
model = WaveletCoefficientUpscaler().to('cuda')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load dataset
dataset = INRDataset(DATASET_PATH + "dataset/raw_data")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, sample in enumerate(data_loader):
        high_res = sample['high_res'].float().to('cuda')  # High-resolution images
        low_res = sample['low_res'].float().to('cuda')   # Low-resolution images

        batch_size = low_res.size(0)
        low_res_coeffs_list = []

        # Apply wavelet transform to each image in the batch
        for b in range(batch_size):
            coeffs = wavelet_transform_image(low_res[b], wavelet='haar', level=1, GPU=True)  # Wavelet transform
            coeffs = torch.tensor(coeffs, dtype=torch.float32).to('cuda')  # Convert to tensor and move to GPU
            low_res_coeffs_list.append(coeffs.unsqueeze(0))  # Add batch dimension

        # Concatenate the transformed coefficients into a single tensor
        low_res_coeffs = torch.cat(low_res_coeffs_list, dim=0)  # Shape: [batch_size, H, W, 12]

        # Pass through the model
        outputs = model(low_res_coeffs)

        output_torch = torch.zeros((batch_size, 256, 256, 3), dtype=torch.float32, device='cuda')

        # Reconstruct the image from model's output coefficients
        for b in range(batch_size):
            output_coeffs = outputs[b].detach().cpu().numpy()  # Shape: [H, W, 12]
            reconstructed_image = reconstruct_image_from_coefficients(output_coeffs, wavelet='haar', GPU=True)
            output_torch[b] = torch.tensor(reconstructed_image, dtype=torch.float32, device='cuda')

        # Calculate loss
        loss = criterion(output_torch, high_res)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Percent loss
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")



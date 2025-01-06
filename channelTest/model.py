import torch
import torch.nn as nn
import numpy as np

class WaveletCoefficientUpscaler(nn.Module):
    def __init__(self):
        super(WaveletCoefficientUpscaler, self).__init__()
        # Define a shared convolutional backbone for the 12 input channels
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Separate paths for upscaling each coefficient group
        self.upsample_approx = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # R_approx, G_approx, B_approx
        self.upsample_horizontal = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # R_horizontal, G_horizontal, B_horizontal
        self.upsample_vertical = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # R_vertical, G_vertical, B_vertical
        self.upsample_diagonal = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # R_diagonal, G_diagonal, B_diagonal

    def forward(self, coeffs):
        # Permute input to match PyTorch's expected shape [batch_size, channels, height, width]
        coeffs = coeffs.permute(0, 3, 1, 2)  # From [batch_size, H, W, C] to [batch_size, C, H, W]
        
        # Shared feature extraction
        x = self.conv(coeffs)
        
        # Separate paths for upscaling each coefficient group
        approx_upscaled = self.upsample_approx(x)
        horizontal_upscaled = self.upsample_horizontal(x)
        vertical_upscaled = self.upsample_vertical(x)
        diagonal_upscaled = self.upsample_diagonal(x)
        
        # Concatenate upscaled components along the channel dimension
        high_res_coeffs = torch.cat([
            approx_upscaled, horizontal_upscaled, vertical_upscaled, diagonal_upscaled
        ], dim=1)  # Shape: [batch_size, 12, H, W]
        
        # Permute back to [batch_size, H, W, 12]
        high_res_coeffs = high_res_coeffs.permute(0, 2, 3, 1)
        
        return high_res_coeffs



if __name__ == "__main__":
    model = WaveletCoefficientUpscaler()
    test = np.zeros((128, 128, 12), dtype=np.float32)
    test = torch.tensor(test, dtype=torch.float32)

    # Add a batch dimension
    test = test.unsqueeze(0)  # Shape: [1, 128, 128, 12]

    print(f"Input shape before model: {test.shape}")
    output = model(test)
    print(f"Output shape from model: {output.shape}")

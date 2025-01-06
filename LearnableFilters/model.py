import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletDecompositionLayer(nn.Module):
    def __init__(self):
        super(WaveletDecompositionLayer, self).__init__()
        # Define the low-pass (L) and high-pass (H) filters for wavelet decomposition
        self.L = nn.Parameter(torch.tensor([[[[1 / 2**0.5, 1 / 2**0.5]]]], dtype=torch.float32), requires_grad=False)
        self.H = nn.Parameter(torch.tensor([[[[1 / 2**0.5, -1 / 2**0.5]]]], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # Apply 2D convolutions with stride=2 (downsample by 2) to perform wavelet decomposition
        LL = F.conv2d(x, self.L, stride=2, padding=1)
        LH = F.conv2d(x, self.H, stride=2, padding=1)
        HL = F.conv2d(x, self.L.transpose(2, 3), stride=2, padding=1)
        HH = F.conv2d(x, self.H.transpose(2, 3), stride=2, padding=1)
        return LL, LH, HL, HH

class WaveletEnhancementLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, padding=1):
        super(WaveletEnhancementLayer, self).__init__()
        self.conv_LL = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_LH = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_HL = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_HH = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, LL, LH, HL, HH):
        # Apply convolutions to each wavelet component
        LL = self.conv_LL(LL)
        LH = self.conv_LH(LH)
        HL = self.conv_HL(HL)
        HH = self.conv_HH(HH)
        return LL, LH, HL, HH

class WaveletReconstructionLayer(nn.Module):
    def __init__(self, L, H):
        super(WaveletReconstructionLayer, self).__init__()
        self.L = L  # Low-pass filter
        self.H = H  # High-pass filter

    def forward(self, LL, LH, HL, HH):
        # Apply transposed convolution (upsampling) to each component
        upsampled_LL = F.conv_transpose2d(LL, self.L, stride=2, padding=1, output_padding=1)
        upsampled_LH = F.conv_transpose2d(LH, self.H, stride=2, padding=1, output_padding=1)
        upsampled_HL = F.conv_transpose2d(HL, self.L.transpose(2, 3), stride=2, padding=1, output_padding=1)
        upsampled_HH = F.conv_transpose2d(HH, self.H.transpose(2, 3), stride=2, padding=1, output_padding=1)
        
        print(upsampled_LL.shape)
        print(upsampled_LH.shape)
        print(upsampled_HL.shape)
        print(upsampled_HH.shape)

        # Combine the upsampled components to reconstruct the image
        reconstructed_image = upsampled_LL + upsampled_LH + upsampled_HL + upsampled_HH
        return reconstructed_image

class WaveletSuperResolutionNet(nn.Module):
    def __init__(self):
        super(WaveletSuperResolutionNet, self).__init__()
        self.decomposition = WaveletDecompositionLayer()
        self.enhancement = WaveletEnhancementLayer()
        self.reconstruction = WaveletReconstructionLayer(self.decomposition.L, self.decomposition.H)

    def forward(self, x):
        # x is expected to have shape (batch_size, in_channels, height, width)
        LL, LH, HL, HH = self.decomposition(x)
        LL, LH, HL, HH = self.enhancement(LL, LH, HL, HH)
        out = self.reconstruction(LL, LH, HL, HH)
        return out

# Example usage
if __name__ == "__main__":
    # Create a tensor of size (batch_size, channels, height, width)
    batch_size = 10
    x = torch.randn(batch_size, 1, 64, 64)  # Shape: (10, 1, 64, 64)
    
    # Create an instance of the network
    model = WaveletSuperResolutionNet()
    
    # Pass the input through the network
    out = model(x)
    
    # Print the output shape
    print("Output shape:", out.shape)

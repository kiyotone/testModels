import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import ptwt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def wavelet_transform_image(image, wavelet='haar', level=1, GPU=True):
    """
    Apply wavelet transform to each color channel of the image and return the wavelet coefficients.
    Returns RGB channel coefficients in the following order:
    [R_approx, G_approx, B_approx, R_horizontal, G_horizontal, B_horizontal,
    R_vertical, G_vertical, B_vertical, R_diagonal, G_diagonal, B_diagonal]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with shape (H, W, 3).")

    coefficients = []
    channels = cv2.split(image)

    for channel in channels:
        if GPU:
            # Convert to PyTorch tensor and move to GPU
            channel = torch.from_numpy(channel.astype(np.float32)).to('cuda')
            coeffs = ptwt.wavedec2(channel, pywt.Wavelet(wavelet), level=level)

            # Process coeffs to remove singleton dimensions
            coeffs = (coeffs[0].squeeze(0), tuple(c.squeeze(0) for c in coeffs[1]))
        else:
            # Use PyWavelets for CPU processing
            coeffs = pywt.wavedec2(channel, wavelet, level=level)

        coefficients.append(coeffs)

    # Initialize output array
    coeff_shape = coefficients[0][1][0].shape  # Get shape of horizontal details
    channel_coeffs = np.zeros((*coeff_shape, 12))

    for i, coeffs in enumerate(coefficients):
        # Assign coefficients to the output array
        channel_coeffs[:, :, i] = coeffs[0].cpu().numpy() if GPU else coeffs[0]
        channel_coeffs[:, :, i + 3] = coeffs[1][0].cpu().numpy() if GPU else coeffs[1][0]
        channel_coeffs[:, :, i + 6] = coeffs[1][1].cpu().numpy() if GPU else coeffs[1][1]
        channel_coeffs[:, :, i + 9] = coeffs[1][2].cpu().numpy() if GPU else coeffs[1][2]

    return channel_coeffs



def normalize_coefficients(coeffs):
    """
    Normalize wavelet coefficients to the range [0, 255] for visualization.
    """
    norm_coeff = np.abs(coeffs)
    norm_coeff = (norm_coeff - np.min(norm_coeff)) / (np.max(norm_coeff) - np.min(norm_coeff)) * 255
    return norm_coeff.astype(np.uint8)

def reconstruct_image_from_coefficients(coefficients, wavelet='haar', GPU=False):
    """
    Reconstruct the original image from wavelet coefficients.
    Takes Input RGB channel coefficients in the following order:
    [R_approx, G_approx, B_approx, R_horizontal, G_horizontal, B_horizontal,
    R_vertical, G_vertical, B_vertical, R_diagonal, G_diagonal, B_diagonal]
    
    Parameters:
    - coefficients: Wavelet coefficients (NumPy array).
    - wavelet: Type of wavelet used ('haar' by default).
    - GPU: Boolean to enable GPU acceleration.
    
    Returns:
    - reconstructed_image: Reconstructed RGB image.
    """
    channels = []

    for i in range(3):  # Process R, G, B channels
        approx = coefficients[:, :, i]
        hor = coefficients[:, :, i+3]
        ver = coefficients[:, :, i+6]
        diag = coefficients[:, :, i+9]

        coeffs = (approx, (hor, ver, diag))

        if GPU:
            # Convert coefficients to PyTorch tensors and move to GPU
            coeffs = (
                torch.from_numpy(coeffs[0]).unsqueeze(0).unsqueeze(0).to('cuda'),  # Shape [1, 1, H, W]
                tuple(torch.from_numpy(c).unsqueeze(0).unsqueeze(0).to('cuda') for c in coeffs[1])
            )

            # Perform inverse wavelet transform on GPU
            channel = ptwt.waverec2(coeffs, pywt.Wavelet(wavelet))[0, 0].cpu().numpy()
        else:
            # Perform inverse wavelet transform on CPU
            channel = pywt.waverec2(coeffs, wavelet)

        # Clip and convert to uint8
        channel = np.clip(channel, 0, 255).astype(np.uint8)
        channels.append(channel)

    # Merge RGB channels
    reconstructed_image = cv2.merge(channels)
    return reconstructed_image



def show_all_channels(image, main_label=None):
    """
    Display each color channel of the image.

    Parameters:
        image (numpy.ndarray): Input image in RGB format.
    """
    patterns = ['Reds', 'Greens', 'Blues']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for x in range(3):        
        axes[x].imshow(image[:, :, x], cmap=patterns[x], vmin=0, vmax=255)
        axes[x].set_title(f'{patterns[x][:-1]} channel')
        axes[x].axis('off')
        
    if main_label:
        fig.suptitle(main_label, fontsize=16)
    plt.show()




def test_code(image , GPU = False):
        # Perform wavelet transform
    wavelet_coeffs = wavelet_transform_image(image, wavelet='haar', level=1, GPU=GPU)
    

    # Normalize and display channels for visualization
    labels = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
    
    
    for i in range(4):
        transformed_image = cv2.merge([normalize_coefficients(wavelet_coeffs[:, :, i]), normalize_coefficients(wavelet_coeffs[:, :, i+3]), normalize_coefficients(wavelet_coeffs[:, :, i+6])])
        show_all_channels(transformed_image, main_label=labels[i])

    # Reconstruct the image
    reconstructed_image = reconstruct_image_from_coefficients(wavelet_coeffs, wavelet='haar', GPU=GPU)
    
    # Calculate PSNR
    mse = np.mean((image - reconstructed_image) ** 2)
    print(f"MSE: {mse:.2f}")
    psnr = 10 * np.log10(255**2 / mse)
    print(f"PSNR: {psnr:.2f} dB")
    

    # Display the original and reconstructed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.axis('off')
    plt.title('Reconstructed Image')

    plt.show()


# Example usage:
if __name__ == "__main__":
    image_path = 'g.jpg'  # Replace with your image path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = wavelet_transform_image(image, wavelet='haar', level=1 , GPU=True)
    
    # print(x)
    
    test_code(image , GPU=True)
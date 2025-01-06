import cv2
import matplotlib.pyplot as plt
import ptwt

# Load and preprocess the image
image_path = 'g.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
image = cv2.resize(image, (256, 256))  # Resize for simplicity

# Perform wavelet decomposition
wavelet = 'haar'
level = 2  # Decomposition level
coeffs = ptwt.wavedec2(image, wavelet=wavelet, level=level)

# Visualize the decomposition
titles = ['Approximation', 'Horizontal Detail', 'Vertical Detail', 'Diagonal Detail']
plt.figure(figsize=(10, 10))
for i, (coeff_type, coeff) in enumerate(coeffs):
    if coeff_type == 'approximation':  # Approximation is stored separately
        plt.subplot(2, 2, 1)
        plt.imshow(coeff, cmap='gray')
        plt.title(f'{titles[0]} (Level {i+1})')
        plt.axis('off')
    else:
        # Visualize detail coefficients
        for j, detail in enumerate(coeff):
            plt.subplot(2, 2, j+2)
            plt.imshow(detail, cmap='gray')
            plt.title(f'{titles[j+1]} (Level {i+1})')
            plt.axis('off')

plt.tight_layout()
plt.show()

# Reconstruct the image from coefficients
reconstructed_image = ptwt.waverec2(coeffs, wavelet=wavelet)

# Compare original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()

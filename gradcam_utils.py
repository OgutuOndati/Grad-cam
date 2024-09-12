# gradcam_utils.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def compute_gradcam(gradients, activations):
    """
    Compute the Grad-CAM heatmap from the captured gradients and activations.
    """
    gradients = gradients[0].mean(dim=[0, 2])
    activations = activations[0]

    # Compute the weighted sum of activations
    weights = gradients
    cam = torch.zeros(activations.shape[1:])

    for i, w in enumerate(weights):
        cam += w * activations[i]

    # Apply ReLU activation
    cam = torch.nn.functional.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()  # Normalize

    return cam.numpy()

def visualize_gradcam(cam, input_image):
    """
    Visualize the Grad-CAM heatmap on the input image.
    """
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    input_image = np.transpose(input_image, (1, 2, 0))  # Assuming input_image is CHW
    overlay = heatmap + np.float32(input_image)
    overlay = overlay / np.max(overlay)
    
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

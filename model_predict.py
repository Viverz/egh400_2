import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# Define constants
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
NUM_CLASSES = 4


# Load the trained model with safe_mode=False
model_path = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\model\hpcmodel\segmentationv4.keras'
model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

# Load the test image(s)
test_image_path = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\patchimage\SYCW0401.jpg'
test_image = imread(test_image_path)
#test_image = resize(test_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(test_image)
predicted_mask = np.argmax(predictions, axis=-1)

# Define color map
color_map = {
    0: [255, 0, 0],    # Class 0 - Ground (red)
    1: [0, 255, 0],    # Class 1 - Tree (green)
    2: [0, 0, 255],    # Class 2 - Sky (blue)
    3: [0, 0, 0],      # Class 3 - Animal (black)
}

# Initialize predicted mask RGB
predicted_mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

# Apply colors for each class
predicted_mask = predicted_mask.squeeze()  # Remove the extra dimension
for class_idx, color in color_map.items():
    predicted_mask_rgb[predicted_mask == class_idx] = color

# Combine original image and predicted mask with transparency
alpha = 0.8  # Set transparency level
overlay = np.copy(test_image[0].astype(np.uint8))
overlay[predicted_mask_rgb != 0] = (1 - alpha) * test_image[0, :, :, :3][predicted_mask_rgb != 0] + alpha * predicted_mask_rgb[predicted_mask_rgb != 0]

# Plot the test image, predicted mask, and overlay
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_image[0].astype(np.uint8))
axes[0].set_title('Test Image')
axes[0].axis('off')

axes[1].imshow(predicted_mask_rgb)
axes[1].set_title('Predicted Mask')
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

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
test_image_path = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\patchimage\SYCW0869.jpg'
#test_image_path = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\newimages\NorthPine\daynorthpine\1792\SYCW0401.jpg'

test_image = imread(test_image_path)
test_image = resize(test_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

# Predictions for each class
p0 = model.predict(np.expand_dims(test_image, axis=0))[0][:, :, 1]


# Predictions for original image
o0 = model.predict(np.expand_dims(test_image, axis=0))[0][:, :, 0]


# Predictions for original image
i0 = model.predict(np.expand_dims(test_image, axis=0))[0][:, :, 2]


# Predictions for original image
u0 = model.predict(np.expand_dims(test_image, axis=0))[0][:, :, 3]


# Combine probabilities for all classes

# Apply thresholds for each class
thresh1 = 0.25
thresh2 = 0.6
thresh3 = 0.35
thresh4 = 0.2

p = (((p0)) > thresh1).astype(np.uint8)
o = (((o0)) > thresh2).astype(np.uint8)
i = (((i0)) > thresh3).astype(np.uint8)
u = (((u0)) > thresh4).astype(np.uint8)

# Combine predictions
combined_preds = np.stack((p, o, i, u), axis=-1)

# Save the prediction array
np.save(os.path.join(save_dir, 'prediction_array.npy'), combined_preds)
# Define color map for each class
color_map = {
    0: [0, 255, 0],    # Class 0
    1: [255, 0, 0],    # Class 1
    2: [0, 0, 255],    # Class 2
    3: [255, 255, 0]   # Class 3

}
# Initialize predicted mask RGB
predicted_mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

# Apply colors for each class
for class_idx, color in color_map.items():
    predicted_mask_rgb[combined_preds[..., class_idx] == 1] = color

# Normalize test_image to [0, 1]
test_image_normalized = test_image.astype(np.float32) / 255.0

# Plot the original image and the overlay of original image with mask
fig, axes = plt.subplots(2, 1, figsize=(4, 8))

# Original image
axes[0].imshow(test_image_normalized)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Overlay of original image with mask
axes[1].imshow(test_image_normalized)
axes[1].imshow(predicted_mask_rgb, alpha=0.6)  # Overlay mask with transparency
axes[1].set_title('Original Image + Prediction')
axes[1].axis('off')

plt.tight_layout()
plt.show()

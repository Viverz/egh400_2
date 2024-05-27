import os
import cv2
import numpy as np
import json

# Define class labels and corresponding group IDs
class_labels = {'ground': 0, 'tree': 1, 'sky': 2, 'animal': 3}

# Define image dimensions (you need to replace these with your actual image dimensions)
image_height = 1792
image_width = 2560

# Path to the directory containing JSON files
json_dir = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\newimages\NorthPine\daynorthpine'

# Path to save the generated mask images
output_dir = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\mask\100masksbinary'

# Iterate through each JSON file in the directory
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        # Load JSON data
        with open(os.path.join(json_dir, filename), 'r') as f:
            data = json.load(f)

        # Create empty mask image
        mask_image = np.zeros((image_height, image_width), dtype=np.uint8)

        # Iterate through each polygon in the JSON data
        for polygon_data in data['shapes']:
            label = polygon_data['label']
            points = np.array(polygon_data['points'], np.int32)

            # Draw polygon on the mask image
            cv2.fillPoly(mask_image, [points], class_labels[label])

        # Save the generated mask image
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        cv2.imwrite(os.path.join(output_dir, mask_filename), mask_image)

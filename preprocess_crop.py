import os
from PIL import Image

def crop_images_in_folder(input_folder, output_folder):
    # Ensure the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".JPG"):
            # Construct full paths
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # Crop the image
            crop_image(input_path, output_path)

def crop_image(input_path, output_path):
    # Open the image
    image = Image.open(input_path)
    
    # Get image dimensions
    width, height = image.size
    
    # Crop parameters
    top_crop = 11
    bottom_crop = 11
    
    # Calculate new dimensions after cropping
    new_height = height - top_crop - bottom_crop
    
    # Crop the image
    cropped_image = image.crop((0, top_crop, width, height - bottom_crop))
    
    # Save the cropped image
    cropped_image.save(output_path)

# Example usage
input_folder = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\newimages\NorthPine\daynorthpine'
output_folder = r'C:\Users\kelvi\OneDrive\Desktop\maskdir\newimages\NorthPine\daynorthpine\1792'
crop_images_in_folder(input_folder, output_folder)

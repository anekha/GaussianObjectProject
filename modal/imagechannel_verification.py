import os
from PIL import Image

# Path to your image folder
image_folder = "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/data/jewelry/ring_one_60frames/images"

# Iterate through all files in the folder
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)

    # Check if the file is an image
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Print details
                print(f"Image: {image_file}")
                print(f" - Mode: {img.mode}")  # Mode gives channel info (e.g., RGB, RGBA)
                print(f" - Size: {img.size}")  # Size gives (width, height)
                print(f" - Number of Channels: {len(img.getbands())}")  # Number of channels
                print()
        except Exception as e:
            print(f"Error reading {image_file}: {e}")

import pandas as pd
from PIL import Image
import os

# Constants
csv_filename = "data\\PMMA_sampled_test.csv"  # Replace with the path to your CSV file
output_directory = "data\\PMMA\\sampled_test"  # Replace with the path to the directory where you want to save the images

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_filename)

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process images
for index, row in df.iterrows():
    image_path = row["filepath"]  # Get the image file path
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            img = img.convert("L")
            # Crop the image
            img_cropped = img.crop(
                (
                    0,
                    678,
                    2452,
                    678 + 767,
                )
            )  # left, upper, right, and lower pixel
            # Resize the cropped image to 256x256 using bilinear interpolation
            img_resized = img_cropped.resize((256, 256), Image.BILINEAR)
            # Save the processed image with the row index as the filename
            img_resized.save(os.path.join(output_directory, f"{index:04d}.png"))
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")

print("Image processing completed.")

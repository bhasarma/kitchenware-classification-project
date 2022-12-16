from PIL import Image
import os
import random

#enter the directory where images are
directory = "/home/bsarma/GitHub/kitchenware-classification/data_version1/images"
os.chdir(directory)
# Print the current working directory to confirm that you are inside the directory
print("current working dir is:"+os.getcwd())



# Open 300 random images in the directory

## Get a list of all image file paths in the directory
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]

## Choose 300 random image file paths
selected_paths = random.sample(image_paths, 300)

# Open the selected images
images = [Image.open(path) for path in selected_paths]


# Calculate the size of each image
image_width, image_height = images[0].size

# Calculate the width and height of the collage
collage_width = image_width * 30
collage_height = image_height * 10

# Create a blank image with the correct dimensions
collage = Image.new('RGB', (collage_width, collage_height))

# Paste each image into the collage
for i in range(10):
    for j in range(30):
        collage.paste(images[i*10 + j], (j*image_width, i*image_height))


# Go up two levels up in the directory tree
os.chdir(os.pardir)
# Print the current working directory to confirm that you are out of the directory
print("current working dir is:"+os.getcwd())

# Save the collage
output_file = "collage.jpg"
collage.save(output_file)
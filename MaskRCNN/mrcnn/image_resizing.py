import os
from PIL import Image
########################################################################
###########################training resizing#####################3333
# Define the directory path and the size
directory = r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\training'
size = (224, 224)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for image formats
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as img:
            img_resized = img.resize(size)
            img_resized.save(image_path)  # Overwrite the original image with resized one
            print(f"Resized {filename}")
    else:
        print(f"Skipped {filename}")
a=5
########################################################################
###########################Validation resizing#####################3333
directory = r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\validation'
size = (224, 224)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for image formats
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as img:
            img_resized = img.resize(size)
            img_resized.save(image_path)  # Overwrite the original image with resized one
            print(f"Resized {filename}")
    else:
        print(f"Skipped {filename}")

b=5
########################################################################
###########################test resizing#####################3333
directory = r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\evaluation'
size = (224, 224)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for image formats
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as img:
            img_resized = img.resize(size)
            img_resized.save(image_path)  # Overwrite the original image with resized one
            print(f"Resized {filename}")
    else:
        print(f"Skipped {filename}")
c=5

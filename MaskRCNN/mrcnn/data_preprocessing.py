import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
import glob
import cv2

dir_base=r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon'

########################################################################
##########################  preprocessing   ########################################
#Image Resizing and normalization:
main_directories = os.listdir(dir_base)
#training
image_paths_training = os.path.join(dir_base, main_directories[1])
image_files = [os.path.join(image_paths_training, f) for f in os.listdir(image_paths_training) if f.endswith(('.png', '.jpg', '.jpeg'))]
preprocessed_images_train = []
for image_path in image_files:
    img = cv2.imread(image_path)  # Use image_path instead of image_files
    
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping...")
        continue

    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    preprocessed_images_train.append(img)

#validation
image_paths_validation = os.path.join(dir_base, main_directories[2])
image_files = [os.path.join(image_paths_validation, f) for f in os.listdir(image_paths_validation) if f.endswith(('.png', '.jpg', '.jpeg'))]
preprocessed_images_val = []
for image_path in image_files:
    img = cv2.imread(image_path)  # Use image_path instead of image_files
    
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping...")
        continue

    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    preprocessed_images_val.append(img)

#test
image_paths_test = os.path.join(dir_base, main_directories[0])
image_files = [os.path.join(image_paths_test, f) for f in os.listdir(image_paths_test) if f.endswith(('.png', '.jpg', '.jpeg'))]
preprocessed_images_test = []
for image_path in image_files:
    img = cv2.imread(image_path)  # Use image_path instead of image_files
    
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping...")
        continue

    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    preprocessed_images_test.append(img)
#######################################################################################
###########################using function################################################
#Just for x_test:
dir_base=r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon'
main_directories = os.listdir(dir_base)
image_paths_test = os.path.join(dir_base, main_directories[0])
image_files = [os.path.join(image_paths_test, f) for f in os.listdir(image_paths_test) if f.endswith(('.png', '.jpg', '.jpeg'))]

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

preprocessed_images_test = []
for image_path in image_files:
    img=preprocess_image(image_path)
    preprocessed_images_test.append(img)






















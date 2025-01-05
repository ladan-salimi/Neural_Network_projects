import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import timeit


start = timeit.timeit()
# Define the directories
base_dir = 'C:/Users/nahas/first_smester/DNN/assignment_three/dataset'
train_dir = os.path.join(base_dir, 'training')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'evaluation')

# Image settings
img_size = (224, 224)  # Resize images to 224x224, you can change this based on your model requirements

# Function to load images and their labels
def load_images_and_labels(folder_dir):
    x_data = []
    y_data = []
    class_names = os.listdir(folder_dir)
    
    for class_name in class_names:
        class_folder = os.path.join(folder_dir, class_name)
        image_filenames = os.listdir(class_folder)
        
        for image_name in image_filenames:
            img_path = os.path.join(class_folder, image_name)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            x_data.append(img_array)
            y_data.append(class_name)  # Use folder name as label
    
    return np.array(x_data), np.array(y_data)

# Load datasets
x_train, y_train = load_images_and_labels(train_dir)
x_valid, y_valid = load_images_and_labels(valid_dir)
x_test, y_test = load_images_and_labels(test_dir)

# Encode the labels (class names) as integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_valid = label_encoder.transform(y_valid)
y_test = label_encoder.transform(y_test)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_train))  # Assuming you have 11 food categories
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

# Normalize image data (optional)
x_train = x_train / 255.0
x_valid = x_valid / 255.0
x_test = x_test / 255.0

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
##################################################################
####################################################################
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import load_img, img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

# Convert y_train from one-hot encoded to integer labels for counting
y_train_int = np.argmax(y_train, axis=1)

# Assuming label_encoder was used to encode classes
food_classes = label_encoder.classes_  # Automatically get class names from the LabelEncoder

# Count occurrences of each class in y_train
counts = [list(y_train_int).count(i) for i in range(len(food_classes))]

# Create the horizontal bar plot
y_pos = np.arange(len(food_classes))
plt.barh(y_pos, counts, align='center', alpha=0.7, color='skyblue')

# Label the plot
plt.yticks(y_pos, food_classes)
plt.xlabel('Counts')
plt.title('Train Data Class Distribution')

# Show the plot
plt.show()

print(f"Training data available in {len(food_classes)} classes")
print(f"Class distribution: {counts}")
#################################################################
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout

def create_lenet_model(input_shape, num_classes):

    model_scratch = Sequential()
    model_scratch.add(Conv2D(32, (3, 3), activation='relu',input_shape =input_shape))
    model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
    model_scratch.add(Dropout(0.25))
        
    model_scratch.add(Conv2D(64, (3, 3), activation='relu'))
    model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
    model_scratch.add(Dropout(0.25))
        
    model_scratch.add(Conv2D(64, (3, 3), activation='relu'))
    model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
    model_scratch.add(Dropout(0.25))
        
    model_scratch.add(Conv2D(128, (3, 3), activation='relu'))
    model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
    model_scratch.add(Dropout(0.25))
        
    model_scratch.add(GlobalAveragePooling2D())
    model_scratch.add(Dense(64, activation='relu'))
    model_scratch.add(Dropout(0.5))
    model_scratch.add(Dense(11, activation='softmax'))
    model_scratch.summary() # Use softmax for multi-class classification
    
    return model_scratch

# Define input shape and number of classes
input_shape = (224, 224, 3)  # Adjust based on your image size
num_classes = 11  # Adjust based on your number of food categories

# Create the model
lenet_model = create_lenet_model(input_shape, num_classes)

# Compile the model
lenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
lenet_model.summary()
############################################################################
# Fit the model
history = lenet_model.fit(x_train, y_train, 
                          validation_data=(x_valid, y_valid),
                          epochs=30,  # Adjust number of epochs as needed
                          batch_size=32)  # Adjust batch size as needed
#########################################################################
fig = plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#################################################################
# Make predictions on the test set
from sklearn.metrics import accuracy_score, confusion_matrix

preds = np.argmax(lenet_model.predict(x_test), axis=1)

# Print accuracy on test data
print("\nAccuracy on Test Data: ", accuracy_score(np.argmax(y_test, axis=1), preds))

# Print number of correctly identified images
print("\nNumber of correctly identified images: ",
      accuracy_score(np.argmax(y_test, axis=1), preds, normalize=False), "\n")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), preds, labels=range(num_classes))
print("Confusion Matrix:\n", conf_matrix)
###########################################################
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the values from the validation dataset
y_pred = lenet_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

end = timeit.timeit()
print(end - start)
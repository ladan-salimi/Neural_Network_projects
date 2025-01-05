import config
import model
import visualize
import cv2
import os
import numpy as np
# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'balloon', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class SimpleConfig(config.Config):
    # Give the configuration a recognizable name
    NAME = "Balloon"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)
############################################################################################
############################  Pre-trained on one test image     ######################################################
######################################################################################################

# load the input image, convert it from BGR to RGB channel
path_test=r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\training\4057490235_2ffdf7d68b_b.jpg'
image = cv2.imread(path_test)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
#image = image.astype(np.float32) / 255.0 
# Perform a forward pass of the network to obtain the results
r = model.detect([image])
r = r[0]

# Visualize the detected objects.
visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
############################################################################################
############################  Pre-trained on three test images     ######################################################
######################################################################################################
import random
import matplotlib.pyplot as plt

# Load the pre-trained Mask R-CNN model
# Ensure 'model' is already loaded with Mask-RCNN
# CLASS_NAMES contains the names of the classes that your model is trained on (e.g. 'balloon')

# Define the path to the test dataset directory
test_dir = r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\evaluation'

# Get a list of all image files in the test directory
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select three images
random_images = random.sample(image_files, 3)

# Create a figure with 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Loop through the selected images and corresponding axes
for idx, (image_path, ax) in enumerate(zip(random_images, axes)):
    # Load the input image, convert from BGR to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image])
    r = r[0]  # Get the results for the first image

    # Display the image with bounding boxes and masks in the current subplot
    visualize.display_instances(image=image, 
                                boxes=r['rois'], 
                                masks=r['masks'], 
                                class_ids=r['class_ids'], 
                                class_names=CLASS_NAMES, 
                                scores=r['scores'], 
                                ax=ax)  # Pass the current axis to visualize the result

    # Set title for each subplot
    ax.set_title(f"Image {idx + 1}")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
############################################################################################
############################  fine tuning    ######################################################
################################################################################################
import config
class TrainConfig(object):
    NAME = "balloon"  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 4#2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH =10 #1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1+1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

import model as modellib

model = modellib.MaskRCNN(mode="training", config=TrainConfig(), model_dir="./logs")
model.load_weights(r"C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\mrcnn\mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])
######################################Dataset preparation##################################
import os
import cv2
import numpy as np
from utils import Dataset
import utils

class BalloonDataset(Dataset):
    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, val, or test
        """
        # Add classes
        self.add_class("balloon", 1, "balloon")

        # Define the subset path (train/validation/test)
        subset_dir = os.path.join(dataset_dir, subset)
        
        # List all image files in the directory
        image_files = [f for f in os.listdir(subset_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        
        for image_id, image_file in enumerate(image_files):
            image_path = os.path.join(subset_dir, image_file)
            
            # Add each image to the dataset with an ID
            self.add_image(
                "balloon",
                image_id=image_id,
                path=image_path,
                # Assuming the mask information is stored elsewhere (implement loading of masks separately)
                width=224,  # Update this with actual image dimensions if needed
                height=224  # Update this with actual image dimensions if needed
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: A 1D array of class IDs of the instance masks.
        """
        # Load the mask for the image (implement based on your dataset)
        info = self.image_info[image_id]
        # Here we are assuming a binary segmentation mask for simplicity
        mask_path = info['path']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 128, 1, 0).astype(bool)
        
        # Return mask and class IDs (1 for balloon)
        return mask[..., np.newaxis], np.array([1], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        return self.image_info[image_id]["path"]


# Define the dataset directory
dataset_dir = r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon'

# Load training dataset
train_dataset = BalloonDataset()
train_dataset.load_balloon(dataset_dir, "training")
train_dataset.prepare()

# Load validation dataset
val_dataset = BalloonDataset()
val_dataset.load_balloon(dataset_dir, "validation")
val_dataset.prepare()
############################################# Train the model####################################
model.train(train_dataset, val_dataset, learning_rate=TrainConfig().LEARNING_RATE, epochs=30, layers='heads')
#############plot for the first epoch################################################
import matplotlib.pyplot as plt
X=[1,2,3,4,5,6,7,8,9,10]
Y=[3.65422024,4.4349,4.1571,3.9412,3.8626,3.7743,3.7358,3.6632,3.6281,3.5895]
plt.title('Loss Over Time')
plt.xlabel('Batches in the first epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
##########################################################################################
############################################plot loss for 30 epoches    ##################################
##########################################################################################
#plot
import matplotlib.pyplot as plt
from keras.callbacks import History

# Define a history object
history = History()

# Train the model and pass the history callback
model.train(train_dataset, val_dataset, 
            learning_rate=TrainConfig().LEARNING_RATE, 
            epochs=30, 
            layers='heads', 
            custom_callbacks=[history])

# Plotting loss after training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
############################################################################################
############################ #visulization on the same image after fine tuning##################
################################################################################################

# load the input image, convert it from BGR to RGB channel
path_test=r'C:\Users\nahas\ladan\academic\Agder\1st smester\DNN\session 4\balloon\training\4057490235_2ffdf7d68b_b.jpg'
image = cv2.imread(path_test)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
#image = image.astype(np.float32) / 255.0 
# Perform a forward pass of the network to obtain the results
r = model.detect([image])
r = r[0]

# Visualize the detected objects.
visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])




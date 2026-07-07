

###########################################################################
# Histogram Equalization
# (Pure OpenCV/NumPy — no TensorFlow involved, kept exactly as-is)



# Histograms


# Display images and histograms


################################################################################
# Edge Detection
# (Pure OpenCV/NumPy — no TensorFlow involved, kept exactly as-is)




#######################################################################
# Data Augmentation
# CONVERTED: tensorflow.keras.preprocessing.image.ImageDataGenerator -> torchvision.transforms
#
# ImageDataGenerator(rotation_range=40, zoom_range=0.2, horizontal_flip=True)
# maps onto the torchvision transforms below:
#   rotation_range=40   -> transforms.RandomRotation(40)
#   zoom_range=0.2      -> transforms.RandomResizedCrop(scale=(0.8, 1.0)) approximates "zoom"
#   horizontal_flip     -> transforms.RandomHorizontalFlip(p=0.5)



########################################################################
# Morphological operations
# (Pure OpenCV/NumPy — no TensorFlow involved, kept exactly as-is)

# Read image
img = cv2.imread(r"C:/Users/nagar/Desktop/DL/Exp_1/Image_1.jfif", 0)

# Convert to binary image

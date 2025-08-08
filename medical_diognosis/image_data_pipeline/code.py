# Import required libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Define the base dataset directory
# This will get the absolute path to the "dataset" folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Step 2: Define sub-directories for training, validation, and test datasets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Step 3: Create ImageDataGenerator instances
# This will apply rescaling and can be used to augment training data if needed
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values between 0 and 1
    rotation_range=15,     # Randomly rotate images
    horizontal_flip=True,  # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Step 4: Create data generators from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical'   # Use 'categorical' for multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do not shuffle test data
)
# Step 7: Print the class indices to verify the mapping
images, labels = next(train_generator)
#Print the shape of the loaded batch
print("Batch shape:",images.shape)
print("Labels shape:", labels.shape)
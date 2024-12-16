from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import hyperparams as hp
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
MOMENTUM = 0.01
IMG_SIZE = (128, 128)  # Resize images to 128x128
PREPROCESS_SAMPLE_SIZE = 400
MAX_WEIGHTS_NUM = 5
BATCH_SIZE = 32
NUM_CLASSES = 28
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

def get_transforms():
    """
    main preprocess like resizing. alll into a single tansform for simplicity
    returns 
        - nothing
    
    """
    transform = transforms.Compose([
        # transforms.Resize(hp.IMG_SIZE),  # resize
        transforms.Resize(IMG_SIZE),  # resize
        transforms.ToTensor(),          # make tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalization
    ])
    return transform

def load_datasets():
    """
    loads training and test sets using ImageFolder
    returns
        - train_dataset: Dataset for training
        - test_dataset: Dataset for validation
    """
    transform = get_transforms()
    # train_dataset = datasets.ImageFolder(root=hp.TRAIN_DIR, transform=transform)
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)

    # test_dataset = datasets.ImageFolder(root=hp.TEST_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    return train_dataset, test_dataset

def get_data_loaders():
    """
    makes dataloaders for easier training using the ImageFolder loads
    returns
        - train_loader: DataLoader for training
        - test_loader: DataLoader for validation
    """
    train_dataset, val_dataset = load_datasets()
    # train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader



# import os
# import random
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# import hyperparams as hp

# class Datasets():
#     """ Class for containing the training and test sets as well as
#     other useful data-related information. Contains the functions
#     for preprocessing.
#     """

#     def __init__(self, data_path, task):
#         pass


#     def calc_mean_and_std(self):
#         pass

#     def standardize(self, img):
#         """ Function for applying standardization to an input image.

#         Arguments:
#             img - numpy array of shape (image size, image size, 3)

#         Returns:
#             img - numpy array of shape (image size, image size, 3)
#         """
#         pass

#     def preprocess_fn(self, img):
#         """ Preprocess function for ImageDataGenerator. """
#         pass

#     def custom_preprocess_fn(self, img):
#         """ Custom preprocess function for ImageDataGenerator. """
#         pass

#     def get_data(self, path, is_vgg, shuffle, augment):
#         """ Returns an image data generator which can be iterated
#         through for images and corresponding class labels.

#         Arguments:
#             path - Filepath of the data being imported, such as
#                    "../data/train" or "../data/test"
#             is_vgg - Boolean value indicating whether VGG preprocessing
#                      should be applied to the images.
#             shuffle - Boolean value indicating whether the data should
#                       be randomly shuffled.
#             augment - Boolean value indicating whether the data should
#                       be augmented or not.

#         Returns:
#             An iterable image-batch generator
#         """

#         if augment:
#             data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=15,
#                 width_shift_range=0.15,
#                 height_shift_range=0.15,
#                 zoom_range=0.2,
#                 brightness_range=[0.8, 1.2],
#                 horizontal_flip=True,
#                 fill_mode='nearest',
#                 preprocessing_function=self.preprocess_fn)

        
#         else:
#             # Don't modify this
#             data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 preprocessing_function=self.preprocess_fn)

#         # # VGG must take images of size 224x224
#         # img_size = 224 if is_vgg else hp.img_size
#         # VGG must take images of size 200x200
#         # img_size = 200 if is_vgg else hp.img_size


#         classes_for_flow = None

#         # Make sure all data generators are aligned in label indices
#         if bool(self.idx_to_class):
#             classes_for_flow = self.classes

#         # Form image data generator from directory structure
#         data_gen = data_gen.flow_from_directory(
#             path,
#             target_size=(hp.img_size, hp.img_size),#img_size
#             class_mode='sparse',
#             batch_size=hp.batch_size,
#             shuffle=shuffle,
#             classes=classes_for_flow)

#         # Setup the dictionaries if not already done
#         if not bool(self.idx_to_class):
#             unordered_classes = []
#             for dir_name in os.listdir(path):
#                 if os.path.isdir(os.path.join(path, dir_name)):
#                     unordered_classes.append(dir_name)

#             for img_class in unordered_classes:
#                 self.idx_to_class[data_gen.class_indices[img_class]] = img_class
#                 self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
#                 self.classes[int(data_gen.class_indices[img_class])] = img_class

#         print(f"Data Generator for path: {path}")
#         print(f"Target size: {hp.img_size}x{hp.img_size}")
#         print(f"Batch size: {hp.batch_size}")
#         print(f"Class mode: {data_gen.class_mode}")
#         print(f"Classes in flow: {classes_for_flow}")
#         print(f"Number of samples in {path}: {data_gen.n}")

#         return data_gen

import os
import glob
import sys

def get_image_paths(data_path, categories, num_train_per_cat):
    """
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    """
    num_categories = len(categories)  # number of scene categories.

    train_image_paths = [None] * (num_categories * num_train_per_cat)
    test_image_paths = [None] * (num_categories)

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.
    train_labels = [None] * (num_categories * num_train_per_cat)
    test_labels = [None] * (num_categories)

    for i, cat in enumerate(categories):
        # change this path
        images = glob.glob(os.path.join("C:/Users/dhlee/OneDrive/Desktop/CS1430_Projects/final-project-Domingo-v/data", 'train', cat, '*.jpg'))


        for j in range(num_train_per_cat):
            train_image_paths[i * num_train_per_cat + j] = images[j]
            train_labels[i * num_train_per_cat + j] = cat
        # change this path
        images = glob.glob(os.path.join("C:/Users/dhlee/OneDrive/Desktop/CS1430_Projects/final-project-Domingo-v/data", 'test', cat, '*.jpg'))
        test_image_paths[i] = images[0]
        test_labels[i] = cat

    return train_image_paths, test_image_paths, train_labels, test_labels

#!/usr/bin/python
import numpy as np
import os
import argparse

from sift_code.helpers import get_image_paths
from sift_code.student import get_tiny_images, build_vocabulary, get_bags_of_words, \
    svm_classify, nearest_neighbor_classify
from sift_code.create_results_webpage import create_results_webpage


def projSceneRecBoW(data_path="C:/Users/dhlee/OneDrive/Desktop/CS1430_Projects/final-project-Domingo-v/data"):
    categories = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
                  'Q','R','S','T','U','V','W','X','Y','Z','nothing','space']

    # This list of shortened category names is used later for visualization.
    abbr_categories = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
                  'Q','R','S','T','U','V','W','X','Y','Z','not','spa']

    # Number of training examples per category to use. Max is 100. For
    # simplicity, we assume this is the number of test cases per category as
    # well.
    num_train_per_cat = 40

    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)


    print('Using bag of words representation for images.')
    vocab_size = 200

    vocab = build_vocabulary(train_image_paths, vocab_size)
    np.save('vocab.npy', vocab)

    train_image_feats = get_bags_of_words(train_image_paths, vocab)

    test_image_feats = get_bags_of_words(test_image_paths, vocab)
        

    predicted_categories, model = svm_classify(train_image_feats, train_labels, test_image_feats)

    create_results_webpage( train_image_paths, \
                            test_image_paths, \
                            train_labels, \
                            test_labels, \
                            categories, \
                            abbr_categories, \
                            predicted_categories)

    return predicted_categories, model, vocab
    
    




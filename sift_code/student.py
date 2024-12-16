import numpy as np
from skimage import io, color, transform
from scipy.stats import mode
from scipy.spatial import distance
from tqdm import tqdm
from skimage.feature import hog
import sift_code.model
from sklearn.cluster import KMeans
import pickle
import imageio

def get_tiny_images(image_paths, extra_credit=False):
    tiny_size = 16
    tiny_images = []
    
    for path in image_paths:
        image = io.imread(path)

        if image.ndim == 3:
            image = color.rgb2gray(image)
        tiny_image = transform.resize(image, (tiny_size, tiny_size), anti_aliasing=True)

        tiny_image = tiny_image.flatten()
        
        # normalize
        tiny_image -= np.mean(tiny_image)
        tiny_image /= (np.linalg.norm(tiny_image) + 1e-10)  
        tiny_images.append(tiny_image)
    return np.array(tiny_images)

def build_vocabulary(image_paths, vocab_size, extra_credit=False):
    '''
    This function samples HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.
    '''
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    feature_vector = True

    all_features = []

    for path in tqdm(image_paths, desc="Building Vocabulary"):
        image = io.imread(path)
        if image.ndim == 3:
            image = color.rgb2gray(image)
        hog_features = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, 
                           block_norm='L2-Hys', feature_vector=feature_vector)

        z = cells_per_block[0]
        hog_blocks = hog_features.reshape(-1, z * z * 9)
        all_features.append(hog_blocks)
    all_features = np.vstack(all_features)
    kmeans = KMeans(n_clusters=vocab_size, max_iter=50, tol=1e-4, random_state=42)
    kmeans.fit(all_features)
    return kmeans.cluster_centers_

def get_bags_of_words(image_paths, vocab, extra_credit=False):
    '''
    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab_size = len(vocab)

    histograms = []

    for path in image_paths:
        image = io.imread(path)
        if image.ndim == 3:
            image = color.rgb2gray(image)

        # get HOG featurs from curr image
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        z = 2 #(two cells per block)
        features = features.reshape(-1, z * z * 9)
        dists = distance.cdist(features, vocab, metric='euclidean')

        closest_words = np.argmin(dists, axis=1)
        histogram, _ = np.histogram(closest_words, bins=np.arange(vocab_size + 1), density=False)

        # normalize
        histogram = histogram / (np.sum(histogram) + 1e-10)
        histograms.append(histogram)
    return np.array(histograms)


def get_bags_of_words_final(orig_image, vocab, extra_credit=False):
    '''
    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab_size = len(vocab)

    histograms = []

    image = imageio.v3.imread(orig_image)

    # Convert to a format compatible with skimage if needed
    image = np.asarray(image)
    image = color.rgb2gray(image)

        # get HOG featurs from curr image
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    z = 2 #(two cells per block)
    features = features.reshape(-1, z * z * 9)
    dists = distance.cdist(features, vocab, metric='euclidean')

    closest_words = np.argmin(dists, axis=1)
    histogram, _ = np.histogram(closest_words, bins=np.arange(vocab_size + 1), density=False)

    histogram = histogram / (np.sum(histogram) + 1e-10)
    histograms.append(histogram)
    
    return np.array(histograms)

def svm_classify(train_image_feats, train_labels, test_image_feats, extra_credit=False):
    '''
    Inputs:
        train_image_feats:  An nxd numpy array, where n is the number of training
                            examples, and d is the image descriptor vector size.
        train_labels:       An n x 1 Python list containing the corresponding ground
                            truth labels for the training data.
        test_image_feats:   An m x d numpy array, where m is the number of test
                            images and d is the image descriptor vector size.

    Outputs:
        An m x 1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    categories = np.unique(train_labels)
    classifiers = {}

    for category in categories:
        binary_labels = np.array([1 if label == category else -1 for label in train_labels])
        svm = sift_code.model.SVM()
        svm.train(train_image_feats, binary_labels)
        classifiers[category] = svm

    with open('classifiers.pkl', 'wb') as f:
        pickle.dump(classifiers, f)

    predictions = []
    for test_feat in test_image_feats:
        scores = {}
        for category, svm in classifiers.items():
            score = svm.model.decision_function([test_feat])
            scores[category] = score[0]

        predicted_category = max(scores, key=scores.get)
        predictions.append(predicted_category)

    return np.array(predictions), svm


def predict_single_image(image, vocab):
    image_histogram = get_bags_of_words_final(image, vocab)

    with open('classifiers.pkl', 'rb') as f:
        classifiers = pickle.load(f)
    
    scores = {}
    for category, svm in classifiers.items():
        score = svm.model.decision_function(image_histogram)
        scores[category] = score[0]
    
    # Predict the category with the highest score
    predicted_category = max(scores, key=scores.get)
    return predicted_category


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, extra_credit=False):
    '''
    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    k = 5
    dists = distance.cdist(test_image_feats, train_image_feats, metric='euclidean')

    predicted_labels = []

    for i in range(test_image_feats.shape[0]):
        closest_indices = np.argsort(dists[i])[:k]
        closest_labels = [train_labels[idx] for idx in closest_indices]
        most_common_label = mode(closest_labels).mode[0]
        predicted_labels.append(most_common_label)

    return np.array(predicted_labels)

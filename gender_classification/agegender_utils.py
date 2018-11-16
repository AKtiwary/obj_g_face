import os
import numpy as np
import cv2 as cv
from utils import *

# image dimension
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
IMAGE_CHANNEL = 3

# number of classes
NUM_CLASSES = 2
classes = ['female', 'male']

# organization of folders
DATASET_DIR = '/media/rdf/01D43136CA749520/datasets/adience/'
IMAGE_DIR = os.path.join(DATASET_DIR, 'aligned')
ATTR_DIR = os.path.join(DATASET_DIR, 'folds')
FILENAME_PREFIX = 'landmark_aligned_face'

MODEL_DIR = 'models'
OUTPUT_DIR = 'output'
PREDS = 'preds'
DATASET_MEAN = os.path.join(OUTPUT_DIR, 'gender_adience_mean.json')
CHECKPOINT_DIR = 'checkpoints'

# resized data and binarized label
IMAGE_HDF5 = os.path.join(OUTPUT_DIR, 'adience.hdf5')

# training info
BATCH_SIZE = 64
NUM_EPOCHS = 110

# face detection cascade
CASCADE_FILE = os.path.join(MODEL_DIR, 'haarcascade_frontalface_alt.xml')

# face alignment
FACIAL_LANDMARK_FILE = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')

def getAttributes(file_dir):
    file_names = []
    genders = []
    ages = []

    raw_data = open(file_dir, 'r').read().rstrip()
    lines = raw_data.split('\n')[1:]

    for line in lines:
        parts = line.split('\t')
        user_id, original_image, face_id, age, gender = parts[:5]

        if age[0] != '(' or gender not in ['f', 'm']:
            continue

        file_names.append(user_id + '/' + '.'.join([FILENAME_PREFIX, face_id, original_image ] ) )
        ages.append(age)
        genders.append(gender)

    return file_names, ages, genders

def getImageData(file_names, verbose=True):
    images = []
    L = len(file_names)
    for i, file_name in enumerate(file_names):
        file_dir = os.path.join(DATASET_DIR, IMAGE_DIR, file_name)
        image = cv.imread(file_dir)
        images.append(image)

        if verbose:
            if (i + 1) % 1000 == 0 or (i + 1) == L:
                LOG(INFO, 'Loaded {:>4d}/{}'.format(i+1, L) )
    return images

########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import imageio
import cv2
from tqdm import tqdm

from keras.utils import Sequence
from keras.applications import imagenet_utils
from keras.models import load_model

from keras.utils import Sequence
from keras.applications import imagenet_utils
from keras.models import load_model


########################################################################################################################
# Function definitions
########################################################################################################################
def load_datasets(datasets=('Joystick nice/', 'WASD/', 'Joysink oversteer/', 'Udacity/'),
                  data_base_path='/home/kalap/Documents/Onlab/Udacity CarND/Datasets/',
                  log_csv_path='driving_log.csv', shuffle = False):
    log_df = pd.DataFrame()
    for dataset in datasets:
        print("Loading", data_base_path + dataset + log_csv_path)
        temp_df = pd.read_csv(data_base_path + dataset + log_csv_path)
        temp_df['center'] = data_base_path + dataset + temp_df['center'].str.strip()
        temp_df['right'] = data_base_path + dataset + temp_df['right'].str.strip()
        temp_df['left'] = data_base_path + dataset + temp_df['left'].str.strip()
        log_df = pd.concat([log_df, temp_df], axis=0, ignore_index=True)

    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    if shuffle:
        log_df = log_df.sample(frac=1).reset_index(drop=True)

    return log_df

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-10, :, :] # remove the sky and the car front
    # return image[65:-15, :, :] # remove the sky and the car front

def separate(data, valid_split=0.2, test_split=0.2, shuffle=True):
    """
    Separate the dataset into 3 different parts. Train, validation and test.
    train_data and test_data sets are 1D numpy arrays.

    returns the train, valid and test data sets
    """

    sum_ = len(data)

    train = data[:int(sum_ * (1 - valid_split - test_split))]
    valid = data[int(sum_ * (1 - valid_split - test_split)):int(sum_ * (1 - test_split))]
    test = data[int(sum_ * (1 - test_split)):]

    return train, valid, test


def show_image_with_steering_cmd(image, steering_cmd, title=""):
    img_height = image.shape[0]
    img_width = image.shape[1]
    plt.imshow(image)
    plt.axis('off')
    l = img_height/2
    x1 = img_width/2
    x2 = img_width/2 + l*np.sin(steering_cmd*25.0/180.0*np.pi)
    y1 = img_height
    y2 = img_height - l*np.cos(steering_cmd*25.0/180.0*np.pi)
    plt.plot([x1, x2], [y1, y2], 'k-')

    x2 = img_width / 2 + l * np.sin(1 * 25.0 / 180.0 * np.pi)
    y2 = img_height - l * np.cos(1 * 25.0 / 180.0 * np.pi)
    plt.plot([x1, x2], [y1, y2], 'k--')

    x2 = img_width / 2 + l * np.sin(-1 * 25.0 / 180.0 * np.pi)
    y2 = img_height - l * np.cos(-1 * 25.0 / 180.0 * np.pi)
    plt.plot([x1, x2], [y1, y2], 'k--')

    plt.title(title)


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.show()

# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, list_IDs, drive_log_df, batch_size=32, dim=(32, 32, 32),
                 n_channels=3, shuffle=True, side_camera_correction=0.15, crop = True):
        # Initialization
        self.dim = dim  # dataset's dimension
        self.batch_size = batch_size  # number of data/epoch
        self.drive_log_df = drive_log_df.copy()  # a dataframe storing driving log and image filenames
        self.list_IDs = list_IDs  # a list containing indexes to be used by the generator
        self.n_channels = n_channels  # number of channels in the photo (RGB)
        self.shuffle = shuffle  # shuffle the data
        self.side_camera_correction = side_camera_correction
        self.crop = True
        # np.random.seed(123)

        self.on_epoch_end()

    def on_epoch_end(self):
        # Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.generate(self.list_IDs_temp)
        return X, Y

    def get_last_batch_ImageIDs(self):
        return self.list_IDs_temp

    def generate(self, tmp_list):
        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(tmp_list):
            camera = np.random.random_integers(0, 2)

            camera = 0

            img_path = self.drive_log_df.iat[ID, camera]
            img = imageio.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

            if self.crop:
                img = crop(img)

            if img.shape[0:2] != self.dim:
                print("Reshaping")
                img = cv2.resize(img, self.dim)

            X[i] = imagenet_utils.preprocess_input(img, mode='tf')
            Y[i] = self.drive_log_df.at[ID, 'steering']

            # if camera == 1:
            #     # left camera
            #     Y[i] += (self.side_camera_correction + np.random.rand()*0.1)
            # elif camera == 2:
            #     # right camera
            #     Y[i] -= (self.side_camera_correction + np.random.rand()*0.1)

            if np.random.random_integers(0, 1) == 1:
                # print("Flippin images")
                X[i] = cv2.flip(X[i], 1)
                Y[i] *= (-1.0)

        return X, Y


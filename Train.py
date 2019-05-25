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

########################################################################################################################
# Function definitions
##################################################################################################
def separate(train_data, valid_split=0.2, test_split=0.2):
    """
    Separate the dataset into 3 different parts. Train, validation and test.
    train_data and test_data sets are 1D numpy arrays.

    returns the train, valid and test data sets
    """

    sum_ = train_data.shape[0]

    train = train_data[:int(sum_ * (1 - valid_split - test_split))]
    valid = train_data[int(sum_ * (1 - valid_split - test_split)):int(sum_ * (1 - test_split))]
    test = train_data[int(sum_ * (1 - test_split)):]

    return train, valid, test


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
    def __init__(self, list_IDs, drive_log_df, img_path_prefix, batch_size=32, dim=(32, 32, 32),
                 n_channels=3, shuffle=True, side_camera_correction=0.2):
        # Initialization
        self.dim = dim  # dataset's dimension
        self.img_prefix = img_path_prefix  # location of the dataset
        self.batch_size = batch_size  # number of data/epoch
        self.drive_log_df = drive_log_df.copy()  # a dataframe storing driving log and image filenames
        self.list_IDs = list_IDs  # a list containing indexes to be used by the generator
        self.n_channels = n_channels  # number of channels in the photo (RGB)
        self.shuffle = shuffle  # shuffle the data
        self.side_camera_correction = side_camera_correction
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
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.generate(list_IDs_temp)

        return X, Y

    def generate(self, tmp_list):
        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(tmp_list):
            camera = np.random.random_integers(0, 2)

            camera = 0

            img_path = self.img_prefix + self.drive_log_df.iat[ID, camera]
            img = imageio.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            X[i] = imagenet_utils.preprocess_input(img, mode='tf')
            if X[i].shape[0:2] != self.dim:
                X[i] = cv2.resize(X[i], self.dim)

            Y[i] = self.drive_log_df.at[ID, 'steering']

            # if camera == 1:
            #     # left camera
            #     Y[i] += self.side_camera_correction + np.random.rand()*0.1
            # elif camera == 2:
            #     # right camera
            #     Y[i] -= self.side_camera_correction + np.random.rand()*0.1

            if np.random.random_integers(0, 1) == 1:
                # print("Flippin images")
                X[i] = cv2.flip(X[i], 1)
                Y[i] *= (-1.0)



        return X, Y


########################################################################################################################
# PARAMETERS
########################################################################################################################
# data_path  = '/home/kalap/Documents/Onlab/Udacity CarND/Bandy data/'
# log_csv_path = 'driving_log_correct.csv'
# data_path  = '/home/kalap/Documents/Onlab/Udacity CarND/Udacity data/'
# log_csv_path = 'driving_log.csv'
data_path  = '/home/andras/AIDriver/Udacity data'
log_csv_path = 'driving_log.csv'

valid_split = 0.15
test_split = 0.15

batch_size = 64

img_dim = (160, 320)
########################################################################################################################
# Load and prepare the data
########################################################################################################################
# log_df = pd.read_csv(data_path + log_csv_path, sep=';', decimal=',')
log_df = pd.read_csv(data_path + log_csv_path)
log_df['center'] = log_df['center'].str.strip()
log_df['right']  = log_df['right'].str.strip()
log_df['left']   = log_df['left'].str.strip()

pd.set_option('max_columns', 7)
pd.set_option('display.width', 160)
print(log_df.head())
print(log_df.describe())
# Split the data
train_img_ids, valid_img_ids, test_img_ids = separate(log_df.index.values, valid_split, test_split)
# np.save("test_img_ids.npy", test_img_ids)

training_generator = DataGenerator(
    train_img_ids,
    log_df,
    data_path,
    batch_size=batch_size,
    dim=img_dim
)

validation_generator = DataGenerator(
    valid_img_ids,
    log_df,
    data_path,
    batch_size=batch_size,
    dim=img_dim
)

gt_img, gt_commands = training_generator.__getitem__(10)
print(gt_img.shape, gt_commands.shape)
print(gt_commands[0])
# plt.imshow(cv2.cvtColor(gt_img[0]*0.5+0.5, cv2.COLOR_RGB2YUV))
plt.imshow(gt_img[0]*0.5+0.5)
plt.axis('off')
plt.show()


steering_cmds = np.empty(20*batch_size)
for idx in tqdm(range(20)):
    gt_img, gt_commands = training_generator.__getitem__(idx)
    steering_cmds[idx*batch_size:(idx+1)*batch_size] = gt_commands.flatten()

print(min(steering_cmds), max(steering_cmds))
plt.hist(steering_cmds, range=(-1, 1), bins=20)
plt.show()


# Build the network
########################################################################################################################
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import regularizers


def NvidiaCNN(input_layer):

    x = BatchNormalization()(input_layer)
    x = Conv2D(filters=3, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=48, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=100, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=50, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=10, activation='tanh')(x)
    x = Dense(units=1, activation='tanh')(x)

    return x


input_layer = Input((*img_dim, 3))
final_layer = NvidiaCNN(input_layer)

model = Model(inputs=input_layer, outputs=final_layer)

opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='mse')
print(model.summary())

########################################################################################################################
# Train the network
########################################################################################################################

patience=3
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='model.hdf5', save_best_only=True, verbose=1)

history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=len(training_generator),
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              callbacks=[checkpointer, early_stopping],
                              verbose=1)

plot_history(history)

########################################################################################################################
# Test the network
########################################################################################################################


test_generator = DataGenerator(
    test_img_ids,
    log_df,
    data_path,
    batch_size=batch_size,
    dim=img_dim
)

preds = model.predict_generator(test_generator, steps=20)
print(preds)
plt.hist(preds)
plt.show()

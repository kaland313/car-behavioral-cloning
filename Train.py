########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import imageio
import cv2
from tqdm import tqdm

from utils import *

from keras.utils import Sequence
from keras.applications import imagenet_utils
from keras.models import load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
########################################################################################################################
# Import function definitions
##################################################################################################
from TrainTestUtils import *

########################################################################################################################
# PARAMETERS
########################################################################################################################
valid_split = 0.15
test_split = 0.15

batch_size = 64

img_dim = (90, 320)
########################################################################################################################
# Load and prepare the data
########################################################################################################################
log_df = load_datasets(shuffle=True)

pd.set_option('max_columns', 7)
pd.set_option('display.width', 350)
print(log_df.head())
print(log_df.describe())
print(log_df.shape)
head_image_paths = log_df['center']
pd.options.display.max_colwidth = 150
print(log_df['center'][:10])


# Split the data
train_img_ids, valid_img_ids, test_img_ids = separate(log_df.index.values, valid_split, test_split)
# plt.hist(log_df.loc[train_img_ids, 'steering'], bins=np.arange(-0.95, 1.0, 0.1))
# plt.show()

#Scale the data
# scaler = preprocessing.StandardScaler(with_mean=False).fit(log_df.loc[train_img_ids, 'steering'].values.reshape((-1, 1)))
# log_df['steering'] = scaler.transform(log_df['steering'].values.reshape((-1, 1)))

filtered_train_img_id = []
for img_id in train_img_ids:
    if abs(log_df.at[img_id, 'steering']) < 0.001:
        if np.random.random_sample() < 0.001:
            filtered_train_img_id.append(img_id)
    else:
        filtered_train_img_id.append(img_id)

train_img_ids = filtered_train_img_id

# plt.hist(log_df.loc[filtered_train_img_id, 'steering'], bins=np.arange(-0.95, 1.0, 0.1))
# plt.show()

np.save("train_img_ids.npy", train_img_ids)
np.save("valid_img_ids.npy", valid_img_ids)
np.save("test_img_ids.npy", test_img_ids)


training_generator = DataGenerator(
    train_img_ids,
    log_df,
    batch_size=batch_size,
    dim=img_dim
)

validation_generator = DataGenerator(
    valid_img_ids,
    log_df,
    batch_size=batch_size,
    dim=img_dim
)

gt_img, gt_commands = training_generator.__getitem__(10)
gt_ID = training_generator.get_last_batch_ImageIDs()

print(gt_img.shape, gt_commands.shape)
print(gt_commands[0])
# plt.imshow(cv2.cvtColor(gt_img[0]*0.5+0.5, cv2.COLOR_RGB2YUV))
# for idx in range(1):
#     filename = log_df.at[gt_ID[idx], 'center']
#     filename = filename[filename.find('Datasets')+9:]
#     show_image_with_steering_cmd(gt_img[idx]*0.5+0.5, gt_commands[idx], filename)


# Look at the distribution of the generator output
steering_cmds = np.empty(50*batch_size)
for idx in tqdm(range(50)):
    gt_img, gt_commands = training_generator.__getitem__(idx)
    steering_cmds[idx*batch_size:(idx+1)*batch_size] = gt_commands.flatten()
print(min(steering_cmds), max(steering_cmds))
# np.set_printoptions(threshold=np.nan)
plt.hist(steering_cmds, bins=np.arange(-0.95, 1.0, 0.1))
plt.show()

# Build the network
########################################################################################################################
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import SGD, Adam
from keras import regularizers


def NvidiaCNN(input_layer):

    x = Conv2D(filters=6, kernel_size=5, strides=(2, 2), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=9, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=12, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=16, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=50, activation='tanh')(x)
    x = Dense(units=25, activation='tanh')(x)
    x = Dense(units=5, activation='tanh')(x)
    x = Dense(units=1, activation='tanh')(x)

    return x


input_layer = Input((*img_dim, 3))
final_layer = NvidiaCNN(input_layer)

# model = load_model("model-fin.hdf5")
model = Model(inputs=input_layer, outputs=final_layer)

# opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=Adam(), loss='mse')
print(model.summary())

########################################################################################################################
# Train the network
########################################################################################################################

early_stopping = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint(filepath='model.hdf5', save_best_only=True, verbose=1)
csv_logger = CSVLogger('training_history.log')
# checkpointer=ModelCheckpoint(filepath='models/model.{epoch:02d}-{val_loss:.3f}.hdf5', save_best_only=False, verbose=1, period=5)

history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=len(training_generator),
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              callbacks=[checkpointer, early_stopping, csv_logger],
                              verbose=1)

model.save('model-fin.hdf5')

plot_history(history)

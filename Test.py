########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

cmap = pl.cm.viridis
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)


# cmap = pl.cm.winter
# my_cmap = cmap(np.arange(cmap.N))
# threshold = 0.6
# my_cmap[:int(cmap.N*threshold), -1] = 0
# my_cmap[int(cmap.N*(1-threshold)):, -1] = 0.75
# my_cmap = ListedColormap(my_cmap)

########################################################################################################################
# Setup tensorflow to run on CPU
########################################################################################################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(device_count = {'GPU': 0}) # Use CPU for the testing
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

########################################################################################################################
# Import function definitions
########################################################################################################################
from TrainTestUtils import *


########################################################################################################################
# PARAMETERS
########################################################################################################################
batch_size = 256
img_dim = (90, 320)
valid_split = 0.15
test_split = 0.15

########################################################################################################################
# Load test data and model
########################################################################################################################
log_df = load_datasets()

train_img_ids = np.load("train_img_ids.npy")
valid_img_ids = np.load("valid_img_ids.npy")
test_img_ids = np.load("test_img_ids.npy")

# model = load_model("models/model.15-0.056.hdf5")
model = load_model("models/model-IIT"
                   ""
                   ".hdf5")
########################################################################################################################
# Test the network
########################################################################################################################

# Prediction histogram
# print(test_preds)
# plt.hist(test_preds)
# plt.show()

# Regplot
# plt.figure()
# sns.regplot(x=test_commands, y=preds.reshape(-1)).set(xlim=(10,30),ylim=(10,30));
# plt.xlabel("Ground truth steering commands")
# plt.ylabel("Predicted steering commands")
# plt.show()

def JointPlotTest(model, img_ids, log_df,img_dim, batch_size=256,title="", scale = False):
    np.random.seed(0)
    test_generator = DataGenerator(
        img_ids,
        log_df,
        batch_size=batch_size,
        dim=img_dim,
        shuffle=False
    )

    scaler = preprocessing.StandardScaler(with_mean=False).fit(
        log_df.loc[train_img_ids, 'steering'].values.reshape((-1, 1)))

    test_img, test_commands = test_generator.__getitem__(0)  # returns batch 0
    if scale:
        test_preds = scaler.inverse_transform([model.predict(test_img)])
    else:
        test_preds = model.predict(test_img)

    # https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid
    test_data = pd.DataFrame(np.transpose([test_commands, test_preds.reshape(-1)]))
    test_data.columns = ["Ground truth steering commands", "Predicted steering commands"]
    grid = sns.JointGrid(x="Ground truth steering commands", y="Predicted steering commands",
                         data=test_data, xlim=(-1, 1), ylim=(-1, 1))
    grid = grid.plot_joint(sns.regplot)
    grid.plot_marginals(sns.distplot, kde=False)
    grid.annotate(metrics.mean_squared_error, template="{stat}: {val:.4f}", stat="$MSE$");
    plt.subplots_adjust(top=0.9)
    grid.fig.suptitle(title + " regression plot")
    print("===============================================================")
    print(title + " metrics")
    print("MSE = ", metrics.mean_squared_error(test_commands,test_preds))
    print("MAE = ", metrics.mean_absolute_error(test_commands, test_preds))
    print("===============================================================")


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

# https://github.com/keras-team/keras/issues/41
# https://github.com/keras-team/keras/issues/1479
def PlotActivations(model, img_ids, log_df, img_dim, batch_size=256):
    np.random.seed(0)
    test_generator = DataGenerator(
        img_ids,
        log_df,
        batch_size=batch_size,
        dim=img_dim,
        shuffle=False
    )

    # input_layer = Input((*img_dim, 3))
    # x = Conv2D(filters=6, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[1].get_weights())(input_layer)
    # x = BatchNormalization()(x)
    # x = Conv2D(filters=9, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[3].get_weights())(x)
    # # x = BatchNormalization()(x)
    # # x = Conv2D(filters=12, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[5].get_weights())(x)
    # # x = BatchNormalization()(x)
    # # x = Conv2D(filters=16, kernel_size=3, activation='relu', weights=model.layers[7].get_weights())(x)
    # model_partial = Model(inputs=input_layer, outputs=x)

    input_layer = Input((*img_dim, 3))
    x = BatchNormalization()(input_layer)
    x = Conv2D(filters=3, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[2].get_weights())(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[4].get_weights())(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[6].get_weights())(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=48, kernel_size=3, activation='relu', weights=model.layers[8].get_weights())(x)
    model_partial = Model(inputs=input_layer, outputs=x)
    model_partial.compile(optimizer=Adam(), loss='mse')

    test_img, test_commands = test_generator.__getitem__(8)
    # activations = model_partial.predict(test_img)
    #
    # # show_image_with_steering_cmd(test_img[0] * 0.5 + 0.5, test_commands[0])
    # plt.subplot(2,1,1)
    # plt.imshow(test_img[0] * 0.5 + 0.5)
    # plt.axis('off')
    # plt.imshow(cv2.resize(np.average(activations[0, :, :, :], axis=2), img_dim[::-1]), cmap=my_cmap)
    # plt.subplot(2, 1, 2)
    # plt.imshow(cv2.resize(np.average(activations[0, :, :, :], axis=2), img_dim[::-1]))
    # plt.axis('off')
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.figure()
    #
    # for idx in range(activations.shape[3]):
    #     plt.subplot(int(activations.shape[3]/2+0.5), 2, idx+1)
    #     plt.imshow(activations[0, :, :, idx])
    #     plt.axis('off')
    #
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Nvidia salient object detection
    layer_outputs = [model_partial.layers[8].output,
                     model_partial.layers[6].output,
                     model_partial.layers[4].output,
                     model_partial.layers[2].output]
    model_partial = Model(inputs=input_layer, outputs=layer_outputs)
    activations = model_partial.predict(test_img)
    print(len(activations), activations[0].shape)

    salient_map = np.ones(activations[0][0, :, :, 0].shape)

    for idx in range(len(activations)):
        print(activations[idx].shape)
        # averaging of feature maps and element wise multiplication with previous layer's salient map
        salient_map = np.multiply(salient_map, np.average(activations[idx][0, :, :, :], axis=2))
        print(salient_map.shape)
        if idx < len(activations)-1:
            salient_map = cv2.resize(salient_map, activations[idx+1][0, :, :, 0].shape[::-1])

    salient_map = cv2.resize(salient_map, img_dim[::-1])
    plt.figure()
    plt.subplot(211)
    plt.imshow(test_img[0] * 0.5 + 0.5)
    image = plt.imshow(salient_map, cmap=my_cmap)
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(salient_map)
    plt.axis('off')

    plt.imsave("img.png", test_img[0] * 0.5 + 0.5)
    plt.imsave("salient-map-transparent.png", salient_map, cmap=my_cmap)
    plt.imsave("salient-map.png", salient_map)

# JointPlotTest(load_model("models/model-IIT.hdf5"), test_img_ids, log_df, img_dim, batch_size, "model-IIT.hdf5", scale=False)
# JointPlotTest(load_model("models/model-IIT-retrained.hdf5"), test_img_ids, log_df, img_dim, batch_size, "model-IIT-retrained.hdf5", scale=False)
# plt.show()


PlotActivations(model, test_img_ids, log_df, img_dim, batch_size)
plt.show()
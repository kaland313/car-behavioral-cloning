########################################################################################################################
# Imports
########################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

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
img_dim = (80, 320)
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
model = load_model("model.hdf5")
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

    input_layer = Input((*img_dim, 3))
    x = Conv2D(filters=6, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[1].get_weights())(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=9, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[3].get_weights())(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=12, kernel_size=5, strides=(2, 2), activation='relu', weights=model.layers[5].get_weights())(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(filters=16, kernel_size=3, activation='relu', weights=model.layers[7].get_weights())(x)
    model_partial = Model(inputs=input_layer, outputs=x)

    # opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_partial.compile(optimizer=Adam(), loss='mse')

    test_img, test_commands = test_generator.__getitem__(1)
    activations = model_partial.predict(test_img)

    print(activations.shape)

    show_image_with_steering_cmd(test_img[0] * 0.5 + 0.5, test_commands[0])
    plt.figure()

    for idx in range(activations.shape[3]):
        plt.subplot(int(activations.shape[3]/2+0.5), 2, idx+1)
        plt.imshow(activations[0, :, :, idx])
        plt.axis('off')

    plt.figure()
    plt.imshow(np.average(activations[0, :, :, :], axis=2))

# JointPlotTest(model, test_img_ids, log_df, img_dim, batch_size, "Test", scale=False)
# JointPlotTest(model, train_img_ids, log_df, img_dim, batch_size, "Training", scale=False)
# plt.show()


PlotActivations(model, test_img_ids, log_df, img_dim, batch_size)
plt.show()
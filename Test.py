########################################################################################################################
# Imports
########################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

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
    plt.subplots_adjust(top=0.9)
    grid.fig.suptitle(title)



JointPlotTest(model, test_img_ids, log_df, img_dim, batch_size, "Test regression plot", scale=True)
JointPlotTest(model, train_img_ids, log_df, img_dim, batch_size, "Training regression plot", scale=True)

plt.show()
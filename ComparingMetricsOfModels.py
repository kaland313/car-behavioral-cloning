
# Imports
########################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from tabulate import tabulate
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
model_worse = load_model("model.hdf5")
model_better = load_model("model-fin.hdf5")
########################################################################################################################
# Test the network
########################################################################################################################

np.random.seed(0)
test_generator = DataGenerator(
    test_img_ids,
    log_df,
    batch_size=batch_size,
    dim=img_dim,
    shuffle=False
)

print("Used/Total number of test images: ", batch_size, "/", len(test_img_ids))

scaler = preprocessing.StandardScaler(with_mean=False).fit(log_df.loc[train_img_ids, 'steering'].values.reshape((-1, 1)))

test_img, true_commands = test_generator.__getitem__(0)  # returns batch 0

pred_commands_worse = model_worse.predict(test_img)
pred_commands_better = model_better.predict(test_img)

# SCALING OR NOT SCALING THE OUTPUT!
scale = False
if scale:
    print("Scaling the outputs: ", "Mean = ", scaler.mean_, "Scale = ", scaler.scale_)
    pred_commands_worse = scaler.inverse_transform([pred_commands_worse])
    pred_commands_better = scaler.inverse_transform([pred_commands_better])
else:
    print("Not scaling the outputs")


table = []
headers = ["Metric", "Better model", "Worse model", "Worse-Better"]

metric_list = [["Mean squared error", metrics.mean_squared_error],
               ["Mean absolute error", metrics.mean_absolute_error],
               # ["Mean squared logarithmic error", metrics.mean_squared_log_error],
               ["Median absolute error", metrics.median_absolute_error],
               ["Explained variance score", metrics.explained_variance_score],
               ["R^2 score", metrics.r2_score]]

for metric in metric_list:
    metric_function = metric[1]
    metric_worse = metric[1](true_commands, pred_commands_worse)
    metric_better = metric[1](true_commands, pred_commands_better)
    # if metric[2]:
    #     better_measured_to_be_better = metric_worse > metric_better
    # else:
    #     better_measured_to_be_better = metric_worse < metric_better
    table.append([metric[0], metric_better, metric_worse, metric_worse-metric_better])
#
print(tabulate(table, headers=headers, tablefmt="psql"))
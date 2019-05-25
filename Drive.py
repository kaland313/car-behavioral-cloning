import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
# import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(device_count = {'GPU': 0}) # Use CPU for the testing
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


########################################################################################################################
# Import function definitions
##################################################################################################
from TrainTestUtils import *
from Paths import *

########################################################################################################################
# Global variables
##################################################################################################
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 30
MIN_SPEED = 10

speed_limit = MAX_SPEED

telemetry_received_t = 0

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global telemetry_received_t
        telemetry_received_t_prev = telemetry_received_t
        telemetry_received_t = time.clock()
        telemetry_period_t = telemetry_received_t-telemetry_received_t_prev

        # The current steering angle of the car
        steering_angle = float(data["steering_angle"].replace(',', '.'))
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"].replace(',', '.'))
        # The current speed of the car
        speed = float(data["speed"].replace(',', '.'))
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            centerlineOffset = float(data["centerlineOffset"].replace(',', '.'))
        except:
            centerlineOffset = 0
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)       # from PIL image to numpy array
            # image = utils.preprocess(image) # apply the preprocessing


            image_cropped = crop(image)

            image_cropped = np.array([image_cropped])       # the model expects 4D array

            # predict the steering angle for the image
            # steering_angle = float(scaler.inverse_transform(model.predict(image, batch_size=1)))
            pred_t_start = time.clock()
            steering_angle = float(model.predict(image_cropped, batch_size=1))
            pred_t = time.clock() - pred_t_start
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            send_control(steering_angle, throttle)
            send_control(steering_angle, throttle)
            cmd_sent_t = time.clock()


            # SimulationImg.set_data(image)
            # # SimulationImg.interpolation = 'bicubic'
            # SimulationFig.canvas.draw()
            # SimulationFig.canvas.flush_events()

            simulation_t= telemetry_period_t*1000 - (cmd_sent_t - telemetry_received_t) * 1000
            print('{:+.3f}% \t {:+.2f}% \t {:+.2f}m/s \t {:+.3f}m \t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}'.format(
                                                                                      steering_angle,
                                                                                      throttle,
                                                                                      speed,
                                                                                      centerlineOffset,
                                                                                      (cmd_sent_t - telemetry_received_t) * 1000,
                                                                                      telemetry_period_t*1000,
                                                                                      simulation_t,
                                                                                      pred_t*1000))

        except Exception as e:
            print(e)
        
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


telemetry_received_t = 0

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__().replace('.',','),
            'throttle': throttle.__str__().replace('.',',')
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    log_df = load_datasets()
    train_img_ids = np.load(trainig_file_path + "train_img_ids.npy")
    valid_img_ids = np.load(trainig_file_path + "valid_img_ids.npy")
    test_img_ids = np.load(trainig_file_path + "test_img_ids.npy")

    scaler = preprocessing.StandardScaler(with_mean=False).fit(log_df.loc[train_img_ids, 'steering'].values.reshape((-1, 1)))

    plt.ion()

    SimulationFig = plt.figure()
    ax = SimulationFig.add_subplot(111)
    SimulationImg = ax.imshow(np.zeros((160,320,3)))

    telemetry_received_t = time.clock()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

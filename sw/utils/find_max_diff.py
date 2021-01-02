"""
Python script to find maximum absolute value of the differenced time series of dataset videos
"""

import os

import numpy as np

import convert_coord as coord
from preprocess_cartesian import clean_x_y_z

PATH_TO_DATA = "/media/demo/DATA/saliency-exploitation/sw/preprocessed_train"
FILE_NAMES = os.listdir(PATH_TO_DATA)

VIDEO_WIDTH = 3840

max_tot = np.zeros((6,))

for file_name in FILE_NAMES:
    data = np.load(os.path.join(PATH_TO_DATA, file_name))
    video_height = 2 * data[0, 1]
    print(data.shape)

    x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
    x_gaze, y_gaze, z_gaze = coord.equirect_to_cart(data[:, 0:2], VIDEO_WIDTH, video_height)

    x_head, x_gaze, y_head, y_gaze, z_head, z_gaze = \
        clean_x_y_z(x_head, x_gaze, y_head, y_gaze, z_head, z_gaze)


    data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                           np.expand_dims(y_head, axis=1),
                           np.expand_dims(z_head, axis=1),
                           np.expand_dims(x_gaze, axis=1),
                           np.expand_dims(y_gaze, axis=1),
                           np.expand_dims(z_gaze, axis=1)))

    data_cart_diff_head = data_cart[1:, :3] - data_cart[:-1, :3]
    data_cart_diff_gaze = data_cart[1:, 3:] - data_cart[:-1, :3]


    data_cart_diff = np.concatenate((data_cart_diff_head, data_cart_diff_gaze), axis=1)
    max_video = np.amax(np.abs(data_cart_diff), axis=0)
    print(max_video)
    max_tot[max_video > max_tot] = max_video[max_video > max_tot]

print(max_tot)

# Output head: np.array([1.67219621, 0.06364631, 1.81424042])
# Output gaze : np.array([1.9677468, 0.81687301, 1.99182478])

# Output: [1.67219621 0.06364631 1.84153001 1.9856328  1.94321741 1.99484275]

# Output without clean: [1.95763439 0.06364631 1.99985595 1.99640642 1.9625715  1.99921348]

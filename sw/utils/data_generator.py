"""
File with data generator used by FFNN+LSTM model to train and evaluate the network
"""

import os

import numpy as np

import utils.convert_coord as coord
from utils.preprocess_cartesian import clean_x_y_z_head

# HARD CODED MAXIMUM DIFFERENCES FOR TRAINING SET!
MAX_DIFF_HEAD = np.array([1.95763439, 0.06364631, 1.99985595])
VIDEO_WIDTH = 3840

def get_XY_head(data, delays_list, n_max_delay, n_lookback, n_delay, inference=False):
    """
    Generate input (X) and output (Y) variables using sliding window technique
    Data: head position
    """
    len_delay = len(delays_list)
    nb_features = data.shape[1]

    X = np.zeros((len(data)-n_max_delay-n_lookback, n_lookback+1, nb_features))
    Y = np.zeros((len(data)-n_max_delay-n_lookback, len_delay+1, nb_features))

    for i in range(0, len(X)):
        for j in range(nb_features):
            X[i, :, j] = data[i:i+n_lookback+1, j]
            Y[i, :, j] = \
                data[i+n_lookback:i+n_lookback+n_max_delay+1:n_delay, j]

    # Normalized differences
    X_diff = X[:, 1:] - X[:, :-1]
    X_norm_diff = X_diff / MAX_DIFF_HEAD
    Y_diff = Y[:, 1:] - Y[:, :-1]
    Y_norm_diff = Y_diff / MAX_DIFF_HEAD

    Y_final_diff = np.zeros(shape=(Y_norm_diff.shape[0],
                                   Y_norm_diff.shape[1]*Y_norm_diff.shape[2]))
    Y_final_diff[:, 0:10] = Y_norm_diff[:, :, 0]
    Y_final_diff[:, 10:20] = Y_norm_diff[:, :, 1]
    Y_final_diff[:, 20:30] = Y_norm_diff[:, :, 2]

    if inference:
        return [X_norm_diff[:, :, 0], X_norm_diff[:, :, 1], X_norm_diff[:, :, 2]], Y

    return [X_norm_diff[:, :, 0], X_norm_diff[:, :, 1], X_norm_diff[:, :, 2]], Y_final_diff

def data_generator_head(path_to_data, file_names, delays_list, n_max_delay, n_lookback,
                        n_delay, inference=False):
    """
    Data generator for FFNN+LSTM model
    X_x: input variable, x coordinate of unit vector v
    X_y: input variable, y coordinate of unit vector v
    X_z: input variable, z coordinate of unit vector v

    Y: output variable, containing
    """
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx]

            data = np.load(os.path.join(path_to_data, file_name))

            video_height = 2 * data[0, 1]

            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            x_head, y_head, z_head = clean_x_y_z_head(x_head, y_head, z_head)

            data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                                   np.expand_dims(y_head, axis=1),
                                   np.expand_dims(z_head, axis=1)))

            [X_x, X_y, X_z], Y = get_XY_head(data_cart, delays_list, n_max_delay, n_lookback,
                                             n_delay, inference=inference)

            X_x = np.expand_dims(X_x, axis=2)
            X_y = np.expand_dims(X_y, axis=2)
            X_z = np.expand_dims(X_z, axis=2)

            yield [X_x, X_y, X_z], Y

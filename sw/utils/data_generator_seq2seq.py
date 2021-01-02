"""
File with all data generators used by Seq2Seq models to train and evaluate the network
"""

import os

import numpy as np

import utils.convert_coord as coord
from utils.preprocess_cartesian import clean_x_y_z_head
from utils.read_write_h5 import read_many_hdf5

# HARD CODED MAXIMUM DIFFERENCES
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

    if inference:
        return X_norm_diff, Y

    return X_norm_diff, Y_norm_diff

def get_XY_head_sal(data, delays_list, n_max_delay, n_lookback, n_delay,
                    file_name=None, inference=False):
    """
    Generate input (X) and output (Y) variables using sliding window technique
    Data: head position, saliency map of FoV of each frame
    """
    len_delay = len(delays_list)

    nb_features = data.shape[1] -1

    final_width_px = 128 # 2560/20
    final_height_px = 72 # 1440/20
    path_to_sal = "/media/demo/DATA/saliency-exploitation/sw/fov_images_train_h5_new/saliency"
    sal_images = read_many_hdf5(path_to_sal, file_name)

    frames_data = data[:, 3].astype(int)

    X_sal_maps = np.zeros((len(data)-n_max_delay-n_lookback,
                           final_height_px, final_width_px, 1),
                          dtype=int)
    X = np.zeros((len(data)-n_max_delay-n_lookback, n_lookback+1, nb_features))
    Y = np.zeros((len(data)-n_max_delay-n_lookback, len_delay+1, nb_features))

    for i in range(0, len(X)):
        X_sal_maps[i] = sal_images[frames_data[i]]
        for j in range(nb_features):
            X[i, :, j] = data[i:i+n_lookback+1, j]
            Y[i, :, j] = data[i+n_lookback:i+n_lookback+n_max_delay+1:n_delay, j]

    # Normalized differences
    X_diff = X[:, 1:] - X[:, :-1]
    X_norm_diff = X_diff / MAX_DIFF_HEAD
    Y_diff = Y[:, 1:] - Y[:, :-1]
    Y_norm_diff = Y_diff / MAX_DIFF_HEAD

    if not inference:
        return [X_norm_diff, X_sal_maps], Y_norm_diff

    return [X_norm_diff, X_sal_maps], Y

def data_generator_head_seq2seq_teacher_forcing(path_to_data, file_names, delays_list,
                                                n_max_delay, n_lookback, n_delay, inference=False,
                                                clean=True):
    """
    Data generator for model trained using the Teacher Forcing method
    Encoder input: Time sequence of past head positions
    Decoder input: During training, the ground truth is fed to the decoder
    """
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx]

            data = np.load(os.path.join(path_to_data, file_name))

            video_height = 2 * data[0, 1]

            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            if clean:
                x_head, y_head, z_head = clean_x_y_z_head(x_head, y_head, z_head)
            data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                                   np.expand_dims(y_head, axis=1),
                                   np.expand_dims(z_head, axis=1)))

            X, Y = get_XY_head(data_cart, delays_list, n_max_delay, n_lookback, n_delay,
                               inference=inference)

            encoder_input = X

            decoder_input = np.zeros(shape=Y.shape)
            decoder_input[:, 0] = np.sum(X[:, -n_delay:], axis=1)
            decoder_input[:, 1:] = Y[:, :-1]

            yield [encoder_input, decoder_input], Y

def data_generator_head_seq2seq_reinject_output(path_to_data, file_names, delays_list, n_max_delay,
                                                n_lookback, n_delay, inference=False, clean=True):
    """
    Data generator for model trained using Output Reinjection in the decoder
    Encoder input: Time sequence of past head positions
    Decoder input: Only one input for decoder, with head position difference in the
                   last 100 ms
    """
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx]

            data = np.load(os.path.join(path_to_data, file_name))

            video_height = 2 * data[0, 1]

            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            if clean:
                x_head, y_head, z_head = clean_x_y_z_head(x_head, y_head, z_head)
            data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                                   np.expand_dims(y_head, axis=1),
                                   np.expand_dims(z_head, axis=1)))

            X, Y = get_XY_head(data_cart, delays_list, n_max_delay, n_lookback, n_delay,
                               inference=inference)

            encoder_input = X

            decoder_input = np.zeros(shape=(Y.shape[0], 1, Y.shape[2]))
            decoder_input[:, 0] = np.sum(X[:, -n_delay:], axis=1)

            yield [encoder_input, decoder_input], Y

def data_generator_head_seq2seq_decoder_zeros(path_to_data, file_names, delays_list,
                                              n_max_delay, n_lookback, n_delay, inference=False,
                                              clean=True, value="zeros"):
    """
    Data generator for model trained using a constant decoder input
    Encoder input: Time sequence of past head positions
    Decoder input: Vector of zeros (or ones)
    """
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx]

            data = np.load(os.path.join(path_to_data, file_name))

            video_height = 2 * data[0, 1]

            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            if clean:
                x_head, y_head, z_head = clean_x_y_z_head(x_head, y_head, z_head)
            data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                                   np.expand_dims(y_head, axis=1),
                                   np.expand_dims(z_head, axis=1)))

            X, Y = get_XY_head(data_cart, delays_list, n_max_delay, n_lookback, n_delay,
                               inference=inference)

            encoder_input = X
            if value == "ones":
                decoder_input = np.ones(shape=Y.shape)
            else:
                decoder_input = np.zeros(shape=Y.shape)

            yield [encoder_input, decoder_input], Y

def data_generator_head_sal_seq2seq_reinject_output(path_to_data, file_names, delays_list,
                                                    n_max_delay, n_lookback, n_delay,
                                                    inference=False, clean=True):
    """
    Data generator for model trained using Output Reinjection
    Encoder input: Time sequence of past head positions
    Decoder input: Only one input for decoder, with the saliency map of the current FoV of
                   the user at time t
    """
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx]
            npy_name = file_name[:-3]+".npy"

            data = np.load(os.path.join(path_to_data, npy_name))

            video_height = 2 * data[0, 1]

            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            frame_ids = data[:, 9]

            if clean:
                x_head, y_head, z_head = clean_x_y_z_head(x_head, y_head, z_head)
            data_cart = np.hstack((np.expand_dims(x_head, axis=1),
                                   np.expand_dims(y_head, axis=1),
                                   np.expand_dims(z_head, axis=1),
                                   np.expand_dims(frame_ids, axis=1)))

            [X, X_sal_maps], Y = get_XY_head_sal(data_cart, delays_list, n_max_delay,
                                                 n_lookback, n_delay, file_name=file_name[:-3],
                                                 inference=inference)

            encoder_input = X

            yield [encoder_input, X_sal_maps], Y

"""
File with late fusion generator (maximum peak coordinate)
"""

import os

import numpy as np

import convert_coord as coord
import preprocess_cartesian
import read_write_h5


def get_XY_head_gaze_sal(data, delays_list, n_max_delay, n_lookback, n_delay):
    len_delay = len(delays_list)
    nb_features = data.shape[1]
    X = np.zeros((len(data) - n_max_delay - n_lookback + 1, nb_features * n_lookback))
    for _ in range(len(delays_list)):
        for i in range(len(X)):
            for j in range(nb_features):
                X[i, j*n_lookback:(j+1)*n_lookback] = data[i:i+n_lookback, j]

    # nb_features-4 because of the saliency + max_sal columns
    Y = np.zeros((len(data) - n_max_delay - n_lookback + 1, 3 * len_delay))
    #print("Y", Y.shape)
    for i in range(len(Y)):
        flag = True
        j_temp = 0
        for j in range(3):
            if flag:
                Y[i, j_temp*len_delay:(j_temp+1)*len_delay] = \
                    data[i+n_delay+n_lookback-1:i+n_max_delay+n_lookback:n_delay, j]
                j_temp += 1
            flag = not flag

    return X, Y


VIDEO_WIDTH = 3840

def data_generator_head_gaze_max_sal(path_to_data, path_to_saliency, file_names,
                                     n_max_delay, n_lookback, n_delay):
    while True:
        perm_file_idxs = np.random.permutation(len(file_names))
        for idx in perm_file_idxs:
            file_name = file_names[idx][:-3]
            #print(file_name)
            data = np.load(os.path.join(path_to_data, file_name)+".npy")
            N = len(data)
            sal_images = read_write_h5.read_many_hdf5(path_to_saliency, file_name)
            #print(sal_images[:5])
            #print("max", np.amax(sal_images))
            max_sal_indexes = np.zeros((len(sal_images), 2))

            for i, elt in enumerate(sal_images[:, :, :, 0]):
                #print(elt.shape)
                index_max = np.argmax(elt, axis=None)
                #print(index_max)
                max_sal_indexes[i] = np.unravel_index(index_max, elt.shape)
                #print(max_sal_indexes[i])

            #print(max_sal_indexes)
            max_sal_indexes -= np.mean(max_sal_indexes, axis=0)
            normed_sal_indexes = max_sal_indexes / np.amax(np.absolute(max_sal_indexes))
            #print(normed_sal_indexes)


            video_height = 2 * data[0, 1]
            frame_ids = data[:, 9]
            x_head, y_head, z_head = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)
            x_gaze, y_gaze, z_gaze = coord.equirect_to_cart(data[:, :2], VIDEO_WIDTH, video_height)
            x_head_clean, x_gaze_clean, y_head_clean, y_gaze_clean, z_head_clean, z_gaze_clean = \
                preprocess_cartesian.clean_x_y_z(x_head, x_gaze, y_head, y_gaze, z_head, z_gaze)

            X_x = np.zeros((N - n_max_delay - n_lookback, n_lookback, 2))
            X_y = np.zeros((N - n_max_delay - n_lookback, n_lookback, 2))
            X_z = np.zeros((N - n_max_delay - n_lookback, n_lookback, 2))
            X_max_sal = np.zeros((N - n_max_delay - n_lookback, n_lookback, 2))
            Y = np.zeros((N - n_max_delay - n_lookback, 30))
            max_index_data = N - n_max_delay - n_lookback
            for i in range(N - n_max_delay - n_lookback):
                frame_id_window = frame_ids[i:i+n_lookback].astype(int)
                #print("window", frame_id_window)
                try:
                    X_max_sal[i] = normed_sal_indexes[frame_id_window]
                    X_x[i, :, 0] = x_head_clean[i:i+n_lookback]
                    X_x[i, :, 1] = x_gaze_clean[i:i+n_lookback]
                    X_y[i, :, 0] = y_head_clean[i:i+n_lookback]
                    X_y[i, :, 1] = y_gaze_clean[i:i+n_lookback]
                    X_z[i, :, 0] = z_head_clean[i:i+n_lookback]
                    X_z[i, :, 1] = z_gaze_clean[i:i+n_lookback]
                    Y[i, 0:10] = \
                        x_head_clean[i+n_delay+n_lookback:i+n_max_delay+n_lookback+1:n_delay]
                    Y[i, 10:20] = \
                        y_head_clean[i+n_delay+n_lookback:i+n_max_delay+n_lookback+1:n_delay]
                    Y[i, 20:30] = \
                        z_head_clean[i+n_delay+n_lookback:i+n_max_delay+n_lookback+1:n_delay]

                except:
                    max_index_data = i
                    break

            yield [X_x[:max_index_data], X_y[:max_index_data], X_z[:max_index_data],
                   X_max_sal[:max_index_data]], Y[:max_index_data]

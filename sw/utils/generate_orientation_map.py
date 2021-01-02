"""
Function to generate orientation map from data
"""

import numpy as np
from scipy import ndimage

def generate_orientation_map(data, height_input, width_input, height_output, width_output):
    """
    Function to generate orientation map from data
    """
    n_data = len(data)
    orientation_map_data = np.zeros((n_data, height_output, width_output, 1))

    x_perc = data[:, 0] / width_input
    y_perc = data[:, 1] / height_input

    x_output = np.floor(x_perc * width_output).astype(int)
    y_output = np.floor(y_perc * height_output).astype(int)
    for i in range(n_data):
        #print(i, height_output, width_output, x_output[i], x_perc[i], y_perc[i], y_output[i])
        orientation_map_data[i, y_output[i], x_output[i], 0] = 1
        orientation_map_data[i, :, :, 0] = ndimage.gaussian_filter(
            orientation_map_data[i, :, :, 0], sigma=3)

    return orientation_map_data

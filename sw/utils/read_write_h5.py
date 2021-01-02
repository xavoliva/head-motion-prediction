"""
File containing read and write functions of h5 files
"""

import os
import numpy as np
import h5py

def read_many_hdf5(directory, file_name):
    """ Reads image from HDF5.
        Parameters:
        directory     directory containing hdf5 files
        file_names    names of files of hdf5 directory
        Returns:
        ----------
        images      images array, (N, height, width, 3) or (N, height, width, 1) to be read
    """
    images = []

    # Open the HDF5 file
    file = h5py.File(os.path.join(directory, file_name)+".h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    return images


def store_many_hdf5(images, directory, file_name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images          images array, (N, height, width, 3) or (N, height, width, 1) to be stored
        directory       directory containing hdf5 files
        file_names      names of files of hdf5 directory
    """

    # Create a new HDF5 file
    file = h5py.File(os.path.join(directory, file_name)+".h5", "w")

    # Create a dataset in the file
    file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )

    file.close()

"""
File defining functions that perform coordinate conversions
"""

import math

import numpy as np


def equirect_to_cart(equirect_data, width_px, height_px):
    """
    Function that converts equirectangular coordinates to cartesian unit vector
    """
    x_eq = equirect_data[:, 0]
    y_eq = equirect_data[:, 1]

    phi_rads = (x_eq *2 * math.pi) / width_px
    theta_rads = (y_eq * math.pi) / height_px

    x_cart = np.multiply(np.sin(theta_rads), np.cos(phi_rads))
    y_cart = np.cos(theta_rads)
    z_cart = np.multiply(np.sin(theta_rads), np.sin(phi_rads))

    return x_cart, y_cart, z_cart

def cart_to_equirect(x_cart, y_cart, z_cart, width_px, height_px):
    """
    Function that converts cartesian unit vector to equirectangular coordinates
    """
    hor_rads = np.arctan2(z_cart, x_cart)
    ver_rads = np.arccos(y_cart)

    x_eq = (hor_rads / (2 * math.pi)) * width_px
    y_eq = (ver_rads / math.pi) * height_px

    ind_negative_y_eq = (y_eq < 0)
    y_eq[ind_negative_y_eq] = -y_eq[ind_negative_y_eq]
    x_eq[ind_negative_y_eq] = x_eq[ind_negative_y_eq] + width_px / 2

    ind_too_big_y_eq = (y_eq >= height_px)
    y_eq[ind_too_big_y_eq] = 2 * height_px - y_eq[ind_too_big_y_eq] -1
    x_eq[ind_too_big_y_eq] = x_eq[ind_too_big_y_eq] + width_px / 2

    ind_negative_x_eq = (x_eq < 0)
    x_eq[ind_negative_x_eq] = width_px + x_eq[ind_negative_x_eq] - 1

    ind_too_big_x_eq = (x_eq > width_px)
    x_eq[ind_too_big_x_eq] -= width_px

    return x_eq, y_eq

def compute_diff(time_sequence):
    """
    First order differencing
    """
    time_sequence_diff = time_sequence[1:] - time_sequence[:-1]
    return time_sequence_diff

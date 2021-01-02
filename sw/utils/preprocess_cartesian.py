"""
File defines functions to perform preprocessing of data:
- apply median filter
- detect and delete outliers
"""

from scipy.signal import medfilt



OUTLIER_THRESHOLD = 0.1

def is_outlier(x_1, x_2, x_3, threshold_value):
    """
    Outlier is defined when sudden data jump from x_1 to x_2
    """
    return abs(x_1 - x_3) <= threshold_value <= abs(x_1 - x_2) \
        and threshold_value <= abs(x_2 - x_3)


def clean_x_y_z(x_head, x_gaze, y_head, y_gaze, z_head, z_gaze):
    """
    Clean head and gaze data by applying median filter and deleting outliers
    """
    # Apply median filter
    kernel_size_head = 251
    x_head = medfilt(x_head, kernel_size_head)
    y_head = medfilt(y_head, kernel_size_head)
    z_head = medfilt(z_head, kernel_size_head)

    kernel_size_gaze = 50+1
    x_gaze = medfilt(x_gaze, kernel_size_gaze)
    y_gaze = medfilt(y_gaze, kernel_size_gaze)
    z_gaze = medfilt(z_gaze, kernel_size_gaze)


    x_head_clean = x_head.copy()
    y_head_clean = y_head.copy()
    z_head_clean = z_head.copy()
    x_gaze_clean = x_gaze.copy()
    y_gaze_clean = y_gaze.copy()
    z_gaze_clean = z_gaze.copy()

    # Avoid data jumps
    for i in range(len(x_gaze)-2):
        if is_outlier(x_head[i], x_head[i+1], x_head[i+2], OUTLIER_THRESHOLD):
            x_head_clean[i+1] = (x_head[i] + x_head[i+2]) / 2
        if is_outlier(y_head[i], y_head[i+1], y_head[i+2], OUTLIER_THRESHOLD):
            y_head_clean[i+1] = (y_head[i] + y_head[i+2]) / 2
        if is_outlier(z_head[i], z_head[i+1], z_head[i+2], OUTLIER_THRESHOLD):
            z_head_clean[i+1] = (z_head[i] + z_head[i+2]) / 2
        if is_outlier(x_gaze[i], x_gaze[i+1], x_gaze[i+2], OUTLIER_THRESHOLD):
            x_gaze_clean[i+1] = (x_gaze[i] + x_gaze[i+2]) / 2
        if is_outlier(y_gaze[i], y_gaze[i+1], y_gaze[i+2], OUTLIER_THRESHOLD):
            y_gaze_clean[i+1] = (y_gaze[i] + y_gaze[i+2]) / 2
        if is_outlier(z_gaze[i], z_gaze[i+1], z_gaze[i+2], OUTLIER_THRESHOLD):
            z_gaze_clean[i+1] = (z_gaze[i] + z_gaze[i+2]) / 2


    return x_head_clean, x_gaze_clean, y_head_clean, y_gaze_clean, z_head_clean, z_gaze_clean

def clean_x_y_z_head(x_head, y_head, z_head):
    """
    Clean head data by applying median filter and deleting outliers
    """
    # Apply median filter
    kernel_size_head = 25
    x_head = medfilt(x_head, kernel_size_head)
    y_head = medfilt(y_head, kernel_size_head)
    z_head = medfilt(z_head, kernel_size_head)

    x_head_clean = x_head.copy()
    y_head_clean = y_head.copy()
    z_head_clean = z_head.copy()

    # Avoid data jumps
    for i in range(len(x_head)-2):
        if is_outlier(x_head[i], x_head[i+1], x_head[i+2], OUTLIER_THRESHOLD):
            x_head_clean[i+1] = (x_head[i] + x_head[i+2]) / 2
        if is_outlier(y_head[i], y_head[i+1], y_head[i+2], OUTLIER_THRESHOLD):
            y_head_clean[i+1] = (y_head[i] + y_head[i+2]) / 2
        if is_outlier(z_head[i], z_head[i+1], z_head[i+2], OUTLIER_THRESHOLD):
            z_head_clean[i+1] = (z_head[i] + z_head[i+2]) / 2

    return x_head_clean, y_head_clean, z_head_clean

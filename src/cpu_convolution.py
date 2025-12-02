# cpu_convolution.py
import cv2
import numpy as np
import time

def cpu_gaussian_blur(gray_img: np.ndarray, kernel: np.ndarray):
    """
    Use OpenCV filter2D as CPU baseline for convolution.
    gray_img: 2D uint8 image
    kernel: 2D float32 kernel (sums to 1)
    """
    t0 = time.perf_counter()
    # OpenCV expects kernel as float32
    out = cv2.filter2D(gray_img, -1, kernel.astype('float32'), borderType=cv2.BORDER_REFLECT)
    t1 = time.perf_counter()
    return out, t1 - t0

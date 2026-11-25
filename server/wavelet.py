"""Wavelet transform used by the original classifier."""

import cv2
import numpy as np
import pywt


def w2d(image, mode="haar", level=1):
    # Keep the original RGB conversion on OpenCV input for artifact compatibility.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    coefficients = pywt.wavedec2(gray, mode, level=level)
    coefficients[0].fill(0)
    reconstructed = pywt.waverec2(coefficients, mode)
    return (reconstructed * 255).astype(np.uint8)

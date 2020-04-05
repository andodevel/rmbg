import cv2 as cv
import numpy as np

from config import get_logger

THRESHOLD = 60
RATIO = 2
KERNEL_SIZE = 3

logger = get_logger()


class CannyResult:

    def __init__(self, image, ratio):
        self.image = image
        self.ratio = ratio


def canny_detect(src, src_gray):
    low_threshold = THRESHOLD
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * RATIO, KERNEL_SIZE)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    return CannyResult(dst, float(np.sum(mask)) / src_gray.size)

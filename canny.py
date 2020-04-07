import cv2 as cv

THRESHOLD = 40
RATIO = 2
KERNEL_SIZE = 2

def apply(src, src_gray):
    low_threshold = THRESHOLD
    img_blur = cv.blur(src_gray, (2, 2))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * RATIO, KERNEL_SIZE)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    return dst

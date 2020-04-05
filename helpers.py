import datetime
import os
import timeit

import cv2
import numpy as np

from config import get_logger

logger = get_logger()


class Timer:
    """Measure time used."""

    def __init__(self, id: str, round_ndigits: int = 2):
        self._id = id
        self._round_ndigits = round_ndigits
        self._start_time = (timeit.default_timer() * 1000)

    def __call__(self) -> float:
        return (timeit.default_timer() * 1000) - self._start_time

    def __str__(self) -> str:
        return f"Time elapsed of `{self._id}`: " + str(
            datetime.timedelta(milliseconds=round(self(), self._round_ndigits)))


def time_exec(id, method, *args, **kwargs):
    _timer = Timer(id)
    result = method(*args, **kwargs)
    logger.info(_timer)
    return result


def load_image(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise Exception(f"Invalid image with path `{imagePath}`")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def rgb_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def apply_image_mask(masks, image, fallback_image):
    """Apply mask to image, masked pixel keep original value,
    otherwise use its gray version"""

    return np.where(
        masks,
        image,
        fallback_image
    ).astype(np.uint8)


def slide_window(array, window=(0,), a_steps=None, w_steps=None, axes=None, to_end=True):
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if array.ndim < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    tmp_a_steps = np.ones_like(orig_shape)
    if a_steps is not None:
        a_steps = np.atleast_1d(a_steps)
        if a_steps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(a_steps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        tmp_a_steps[-len(a_steps):] = a_steps

        if np.any(a_steps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    a_steps = tmp_a_steps

    tmp_w_steps = np.ones_like(window)
    if w_steps is not None:
        w_steps = np.atleast_1d(w_steps)
        if w_steps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(w_steps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        tmp_w_steps[:] = w_steps
        tmp_w_steps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    w_steps = tmp_w_steps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * w_steps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += w_steps - _window * w_steps
    new_shape = (new_shape + a_steps - 1) // a_steps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= a_steps
    new_strides = array.strides[-len(window):] * w_steps

    # The full new shape and strides:
    if to_end:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def simple_correlate(array, kernel):
    pass


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if (width is not None and width > w) or (height is not None and height > h):
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


# TODO: replace by checking magic numbers of file format
def is_image_file(file_path):
    _, filename = os.path.split(file_path)
    return os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

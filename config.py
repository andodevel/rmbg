import logging

# Object take more than main_object_ratio of the image is considered main object.
main_object_ratio = 0.02

color_summary = {'n_clusters': 10}

fuzzy_kernel_ratio = 0.05  # 5% of image

interested_classes = []

visual_debug_enabled = False

debug_enabled = False

force_gpu_disabled = False



# ======== Functions to resolve configuration.

def use_cpu():
    global force_gpu_disabled
    force_gpu_disabled = True


def cal_fuzzy_kernel_size(image):
    width = image.shape[0]
    height = image.shape[1]
    ref_size = width if width < height else height
    # Kernel should use odd size
    size = (int(ref_size * fuzzy_kernel_ratio / 2) * 2) + 1

    return size

def enable_debug():
    global debug_enabled
    debug_enabled = True
    global visual_debug_enabled
    visual_debug_enabled = True

def get_logger():
    logger_name = 'debug' if debug_enabled else 'info'
    return logging.getLogger(logger_name)

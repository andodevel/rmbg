import logging.config
import os
import sys

import keras
import numpy as np
import tensorflow as tf
import yaml
from distutils.version import LooseVersion

import config as cfg
import constants as cts


def check_requirements():
    # Check 3rd party libs requirement
    assert LooseVersion(tf.__version__) >= LooseVersion('2.1')
    assert LooseVersion(keras.__version__) >= LooseVersion('2.3.1')


def enable_gpu_memoy_growth():
    # Work-around issue: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


def config_logging():
    with open('log.yaml', 'r') as f:
        os.makedirs(os.path.dirname("logs/"), exist_ok=True)
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def do_bootstrap():
    check_requirements()

    # To absolutely turn on/off DEBUG mode, we need to update both log.yaml and config file
    config_logging()

    if cfg.debug_enabled:
        np.set_printoptions(threshold=sys.maxsize)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = cts.TFLogLevel.ERROR.value
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = cts.TFLogLevel.OFF.value

    if cfg.force_gpu_disabled:
        os.environ['CUDA_VISIBLE_DEVICES'] = cts.CPU_DEVICE
    else:
        enable_gpu_memoy_growth()

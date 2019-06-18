from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.GPUS = '0, 1, 2, 3'
config.OUTPUT_DIR = 'black2rgb'
config.LOG_DIR = 'log'
config.NUM_WORKERS = 8

# cudnn
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# train
config.TRAIN = edict()
config.TRAIN.START_EPOCH = 0
config.TRAIN.END_EPOCH = 30
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.RESUME = False
config.TRAIN.SHUFFLE = True
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.LR = 2e-4
config.TRAIN.LR_STEP = [15, 25]
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.BETAS = (0.5, 0.999)
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.NESTEROV = False
config.TRAIN.WD = 1e-5
config.TRAIN.SAMPLE_INTERVAL = 500
config.TRAIN.CHECKPOINT_INTERVAL = 1
config.TRAIN.PRINT_FREQ = 100

# test
config.TEST = edict()
config.TEST.MODEL_FILE = ''
config.TEST.SHUFFLE = False
config.TEST.BATCH_SIZE = 32
config.TEST.PRINT_FREQ = 100

# data
config.DATA = edict()
config.DATA.IMAGE_SHAPE = [3, 32, 32]
config.DATA.TRAIN_DATASET = '../dataset/images_train'
config.DATA.TRAIN_DATASET_B = '../dataset/images_train_black'
config.DATA.TEST_DATASET = '../dataset/images_test'
config.DATA.TEST_DATASET_B = '../dataset/images_test_black'

# loss
config.LOSS = edict()
config.LOSS.CYCLE_WEIGHT = 10
config.LOSS.IDENTITY_WEIGHT = 5

# network
config.NETWORK = edict()
config.NETWORK.NUM_RES_BLOCKS = 2


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
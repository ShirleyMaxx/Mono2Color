import argparse
import os
import numpy as np
import itertools
import time
import datetime
import sys
import pprint
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import _init_paths
import models
from dataset import ImageDataset
from utils import LambdaLR, ReplayBuffer
from utils import get_optimizer, get_lr_scheduler, create_logger
from utils import save_checkpoint, load_checkpoint
from config import config
from config import update_config
from function import train, test


from tensorboardX import SummaryWriter


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    gpus = [int(i) for i in config.GPUS.split(',')]

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')

    # initialize generator and discriminator
    G_AB=eval('models.cyclegan.get_generator')(config.DATA.IMAGE_SHAPE,config.NETWORK.NUM_RES_BLOCKS)
    G_BA=eval('models.cyclegan.get_generator')(config.DATA.IMAGE_SHAPE,config.NETWORK.NUM_RES_BLOCKS)
    D_A=eval('models.cyclegan.get_discriminator')(config.DATA.IMAGE_SHAPE)
    D_B=eval('models.cyclegan.get_discriminator')(config.DATA.IMAGE_SHAPE)
    #logger.info(pprint.pformat(G_AB))
    #logger.info(pprint.pformat(D_A))

    # multi-gpus

    model_dict = {}
    model_dict['G_AB']=torch.nn.DataParallel(G_AB,device_ids=gpus).cuda()
    model_dict['G_BA']=torch.nn.DataParallel(G_BA,device_ids=gpus).cuda()
    model_dict['D_A']= torch.nn.DataParallel(D_A,device_ids=gpus).cuda()
    model_dict['D_B']= torch.nn.DataParallel(D_B,device_ids=gpus).cuda()

    # loss functions
    criterion_dict={}
    criterion_dict['GAN']=torch.nn.MSELoss().cuda()
    criterion_dict['cycle']=torch.nn.L1Loss().cuda()
    criterion_dict['identity']=torch.nn.L1Loss().cuda()

    if config.TEST.MODEL_FILE:
        _, model_dict, _ = load_checkpoint(model_dict, {}, final_output_dir, is_train=False)
    else:
        logger.info('[error] no model file specified')
        assert 0

    #Buffers of previously generated samples
    fake_A_buffer=ReplayBuffer()
    fake_B_buffer=ReplayBuffer()

    # Image transformations
    transforms_ = [
        #transforms.Resize(int(config.img_height * 1.12), Image.BICUBIC),
        #transforms.RandomCrop((config.img_height, config.img_width)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Dataset
    logger.info('=> loading testing dataset...')

    test_dataset = ImageDataset(
        config.DATA.TEST_DATASET_B,
        config.DATA.TEST_DATASET,
        transforms_=transforms_,
        mode='test')

    # Test data loader
    test_dataloader=DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test(config, model_dict, test_dataloader, criterion_dict, final_output_dir)
    logger.info('=> finished testing, saving generated images to {}'.format(final_output_dir+'/images'))



if __name__ == '__main__':
    main()
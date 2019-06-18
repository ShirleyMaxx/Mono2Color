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
        config, args.cfg, 'train')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

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

    # optimizers
    optimizer_dict = {}
    optimizer_dict['G'] = get_optimizer(config, itertools.chain(G_AB.parameters(),G_BA.parameters()))
    optimizer_dict['D_A'] = get_optimizer(config, D_A.parameters())
    optimizer_dict['D_B'] = get_optimizer(config, D_B.parameters())

    start_epoch = config.TRAIN.START_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, model_dict, optimizer_dict = load_checkpoint(model_dict, optimizer_dict, final_output_dir)

    # learning rate schedulers
    lr_scheduler_dict = {}
    lr_scheduler_dict['G']=get_lr_scheduler(config, optimizer_dict['G'])
    lr_scheduler_dict['D_A']=get_lr_scheduler(config, optimizer_dict['D_A'])
    lr_scheduler_dict['D_B']=get_lr_scheduler(config, optimizer_dict['D_B'])
    for steps in range(start_epoch):
        for lr_scheduler in lr_scheduler_dict.values():
            lr_scheduler.step()

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
    logger.info('=> loading train and testing dataset...')

    train_dataset = ImageDataset(
        config.DATA.TRAIN_DATASET_B,
        config.DATA.TRAIN_DATASET,
        transforms_=transforms_)
    test_dataset = ImageDataset(
        config.DATA.TEST_DATASET_B,
        config.DATA.TEST_DATASET,
        transforms_=transforms_,
        mode='test')
    # Training data loader
    train_dataloader=DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.NUM_WORKERS
    )
    # Test data loader
    test_dataloader=DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):

        train(config, epoch, model_dict, fake_A_buffer, fake_B_buffer, train_dataloader, criterion_dict, optimizer_dict, lr_scheduler_dict, writer_dict)

        test(config, model_dict, test_dataloader, criterion_dict, final_output_dir)

        for lr_scheduler in lr_scheduler_dict.values():
            lr_scheduler.step()

        if config.TRAIN.CHECKPOINT_INTERVAL !=-1 and epoch%config.TRAIN.CHECKPOINT_INTERVAL==0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': 'cyclegan',
                'state_dict_G_AB': model_dict['G_AB'].module.state_dict(),
                'state_dict_G_BA': model_dict['G_BA'].module.state_dict(),
                'state_dict_D_A': model_dict['D_A'].module.state_dict(),
                'state_dict_D_B': model_dict['D_B'].module.state_dict(),
                'optimizer_G': optimizer_dict['G'].state_dict(),
                'optimizer_D_A': optimizer_dict['D_A'].state_dict(),
                'optimizer_D_B': optimizer_dict['D_B'].state_dict(),
            }, final_output_dir)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
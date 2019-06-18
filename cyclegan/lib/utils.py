import os
import random
import logging
import time
from pathlib import Path
from torch.autograd import Variable
import torch
import torch.optim as optim

#a buffer of 50 for discriminator
class ReplayBuffer:
    def __init__(self,max_size=50):
        assert max_size>0, "empty buffer or trying to create a black hole. Be careful."
        self.max_size=max_size
        self.data=[]

    def push_and_pop(self,data):
        to_return=[]
        for element in data.data:
            element=torch.unsqueeze(element,0)
            if len(self.data)<self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1)>0.5:
                    i=random.randint(0,self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i]=element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

#decay after 100 epochs
class LambdaLR:
    def __init__(self,n_epochs,offset,decay_start_epoch):
        assert(n_epochs-decay_start_epoch)>0,"Dacay must start before the training session ends!"
        self.n_epochs=n_epochs
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def step(self,epoch):
        return 1.0-max(0,epoch+self.offset-self.decay_start_epoch)/(self.n_epochs-self.decay_start_epoch)

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    model = 'cyclegan'
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / model / cfg_name
    final_output_image_dir = final_output_dir / 'images'

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_image_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_optimizer(cfg, paras):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            paras,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            paras,
            lr=cfg.TRAIN.LR,
            betas=cfg.TRAIN.BETAS
        )

    return optimizer

def get_lr_scheduler(cfg, optimizer):
    #return optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=LambdaLR(cfg.TRAIN.END_EPOCH, cfg.TRAIN.START_EPOCH, cfg.TRAIN.LR_STEP).step)
    return optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR)

def load_checkpoint(model_dict, optimizer_dict, output_dir, filename='checkpoint.pth.tar', is_train=True):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        for key in model_dict.keys():
            model_dict[key].module.load_state_dict(checkpoint['state_dict_'+key])
        if is_train:
            for key in optimizer_dict.keys():
                optimizer_dict[key].load_state_dict(checkpoint['optimizer_'+key])
        print('=> loading checkpoint {} (epoch {})'
              .format(file, start_epoch))
        return start_epoch, model_dict, optimizer_dict
    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model_dict, optimizer_dict


def save_checkpoint(states, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
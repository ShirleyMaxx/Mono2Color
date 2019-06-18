import os
import numpy as np
import itertools
import time
import datetime
import logging
import sys
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def train(config, epoch, model_dict, fake_A_buffer, fake_B_buffer, dataloader, criterion_dict, optimizer_dict, lr_scheduler_dict, writer_dict):

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_G = AverageMeter()
    losses_D = AverageMeter()
    losses_CYC = AverageMeter()
    losses_ADV = AverageMeter()

    for model in model_dict.values():
        model.train()

    end=time.time()
    Tensor = torch.cuda.FloatTensor
    for i, batch in enumerate(dataloader):

        #Set model input
        real_A=Variable(batch["A"].type(Tensor))
        real_B=Variable(batch["B"].type(Tensor))

        #Adversarial ground truths
        valid=Variable(Tensor(np.ones((real_A.size(0),*model_dict['D_A'].module.output_shape))),requires_grad=False)
        fake=Variable(Tensor(np.zeros((real_A.size(0),*model_dict['D_A'].module.output_shape))),requires_grad=False)

        ######Train Generators#####

        #GAN Loss
        fake_B=model_dict['G_AB'](real_A)
        loss_GAN_AB=criterion_dict['GAN'](model_dict['D_B'](fake_B),valid)
        fake_A=model_dict['G_BA'](real_B)
        loss_GAN_BA=criterion_dict['GAN'](model_dict['D_A'](fake_A),valid)

        loss_GAN=(loss_GAN_AB+loss_GAN_BA)/2

        #Cycle Loss
        reconv_A=model_dict['G_BA'](fake_B)
        loss_cycle_A=criterion_dict['cycle'](reconv_A,real_A)
        reconv_B=model_dict['G_AB'](fake_A)
        loss_cycle_B=criterion_dict['cycle'](reconv_B,real_B)

        loss_cycle=(loss_cycle_A+loss_cycle_B)/2

        #loss identity
        loss_id=(criterion_dict['identity'](fake_A,real_A)+criterion_dict['identity'](fake_B,real_B))/2

        #Total Generator Loss
        loss_G=loss_GAN+config.LOSS.CYCLE_WEIGHT*loss_cycle+config.LOSS.IDENTITY_WEIGHT*loss_id

        optimizer_dict['G'].zero_grad()
        loss_G.backward()
        optimizer_dict['G'].step()
        losses_G.update(loss_G.item(), real_A.size(0))
        losses_CYC.update(loss_cycle.item(), real_A.size(0))
        losses_ADV.update(loss_GAN.item(), real_A.size(0))


        ####Train Discriminator####
        #A discriminator
        #Real loss
        loss_real=criterion_dict['GAN'](model_dict['D_A'](real_A),valid)
        #fake loss(on batch of previously generated samples)
        fake_A_=fake_A_buffer.push_and_pop(fake_A)
        loss_fake=criterion_dict['GAN'](model_dict['D_A'](fake_A_.detach()),fake)
        #Total loss
        loss_D_A=(loss_real+loss_fake)/2

        optimizer_dict['D_A'].zero_grad()
        loss_D_A.backward()
        optimizer_dict['D_A'].step()

        #B discriminator
        #Real loss
        loss_real=criterion_dict['GAN'](model_dict['D_B'](real_B),valid)
        #fake loss(on batch of previously generated samples)
        fake_B_=fake_B_buffer.push_and_pop(fake_B)
        loss_fake=criterion_dict['GAN'](model_dict['D_B'](fake_B_.detach()),fake)
        #Total loss
        loss_D_B=(loss_real+loss_fake)/2

        optimizer_dict['D_B'].zero_grad()
        loss_D_B.backward()
        optimizer_dict['D_B'].step()

        loss_D=(loss_D_A+loss_D_B)/2
        losses_D.update(loss_D.item(), real_A.size(0))

        #Determine approximate time left
        batches_done=epoch*len(dataloader)+i
        batches_left=config.TRAIN.END_EPOCH*len(dataloader)-batches_done
        time_left=datetime.timedelta(seconds=batches_left*(time.time()-end))
        batch_time.update(time.time()-end)
        end=time.time()

        if i % config.TRAIN.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}/{1}][{2}/{3}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'loss_G {loss_G.val:.5f} ({loss_G.avg:.5f})\t' \
                  'loss_D {loss_D.val:.5f} ({loss_D.avg:.5f})\t' \
                  'loss_CYC {loss_CYC.val:.5f} ({loss_CYC.avg:.5f})\t' \
                  'loss_ADV {loss_ADV.val:.5f} ({loss_ADV.avg:.5f})\t' \
                  'Time Left: {4}\t'.format(
                    epoch, config.TRAIN.END_EPOCH, i, len(dataloader), time_left, batch_time=batch_time, loss_G=losses_G, loss_D=losses_D, loss_CYC=losses_CYC, loss_ADV=losses_ADV)
            logger.info(msg)

            writer=writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss/Loss_G', losses_G.val, global_steps)
            writer.add_scalar('train_loss/Loss_D', losses_D.val, global_steps)
            writer.add_scalar('train_loss/Loss_CYC', losses_CYC.val, global_steps)
            writer.add_scalar('train_loss/Loss_ADV', losses_ADV.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def test(config, model_dict, dataloader, criterion_dict, final_output_dir):
    for model in model_dict.values():
        model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        end=time.time()
        Tensor = torch.cuda.FloatTensor
        for i, batch in enumerate(dataloader):
            real_A=Variable(batch["A"].type(Tensor))
            real_B=Variable(batch["B"].type(Tensor))

            fake_B=model_dict['G_AB'](real_A)
            fake_A=model_dict['G_BA'](real_B)

            ncols = real_A.shape[0]     # [N,3,32,32]

            #Arange images along x axis
            real_A=make_grid(real_A,nrow=ncols,normalize=True)
            real_B=make_grid(real_B,nrow=ncols,normalize=True)
            fake_A=make_grid(fake_A,nrow=ncols,normalize=True)
            fake_B=make_grid(fake_B,nrow=ncols,normalize=True)

            #Arange images along y axis
            image_grid = torch.cat((real_A,fake_B,real_B,fake_A), 1)
            save_image(image_grid,"%s/images/%s.png"%(final_output_dir,i+1),normalize=False)

            if i % config.TEST.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Batch: [{0}/{1}]\t'.format(i, len(dataloader))
                logger.info(msg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 22:44:01 2019

@author: molly
"""

import torch.nn as nn


#init weight
def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
        if hasattr(m,"bias") and m.bias is not None:
            nn.init.constant_(m.bias.data,0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data, 0.0)


###########resnet###############
class ResidualBLock(nn.Module):
    def __init__(self,in_features):
        super(ResidualBLock,self).__init__()

        #(3,h,w)
        self.block=nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features,in_features,3),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features,in_features,3),
        nn.InstanceNorm2d(in_features)
        )

    def forward(self,x):
        return  x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self,input_shape,num_residual_blocks):
        super(GeneratorResNet,self).__init__()
        #input(3,32,32)
        channels=input_shape[0]

        #initial conv block
        out_features=64
        model=[
        nn.ReflectionPad2d(channels), #(3,35,35)
        nn.Conv2d(channels,out_features,7),  #(64,32,32)
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
        ]
        in_features=out_features

        #Downsampling
        for _ in range(2):
            out_features*=2
            model +=[
            nn.Conv2d(in_features,out_features,3,stride=2,padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            ]
            in_features=out_features
            #output(128,16,16)
            #output(256,8,8)

        #Residual blocks
        for _ in range(num_residual_blocks):
            model+=[ResidualBLock(out_features)]
            #output(256,8,8)

        #Upsampling
        for _ in range(2):
            out_features //=2
            model+=[
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features,out_features,3,stride=1,padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
            ]
            in_features=out_features
            #output(128,16,16)
            #output(64,32,32)

        #Output layer
        model+=[
        nn.ReflectionPad2d(channels),
        nn.Conv2d(out_features,channels,7),
        nn.Tanh()
        ]#(3,32,32)

        self.model=nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)


###########Discriminator###############
class Discriminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator,self).__init__()

        channels,height,width=input_shape
        
        #output shape of PatchGAN
        self.output_shape=(1,height//2**4,width//2**4)

        #return downsampling layers of each discriminator_block
        def discriminator_block(in_filters,out_filters,normalize=True):
            layers=[nn.Conv2d(in_filters,out_filters,4,stride=2,padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        #(3,32,32)
        self.model=nn.Sequential(
        *discriminator_block(channels,64,normalize=False),#(64,16,16)
        *discriminator_block(64,128),#(128,8,8)
        *discriminator_block(128,256),#(256,4,4)
        *discriminator_block(256,512),#(512,2,2)
        nn.ZeroPad2d((1,0,1,0)), #(512,3,3)
        nn.Conv2d(512,1,4,padding=1)#(1,2,2)
        )

    def forward(self,img):
        return self.model(img)

def get_generator(input_shape, num_residual_blocks, is_train=True):
    generator = GeneratorResNet(input_shape, num_residual_blocks)
    if is_train:
        generator.apply(weight_init_normal)
    return generator

def get_discriminator(input_shape, is_train=True):
    discriminator = Discriminator(input_shape)
    if is_train:
        discriminator.apply(weight_init_normal)
    return discriminator
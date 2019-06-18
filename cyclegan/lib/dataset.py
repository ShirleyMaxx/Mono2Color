#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:36:19 2019

@author: molly
"""

import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def to_rgb(image):
    rgb_image=Image.new("RGB",image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self,root1,root2,transforms_=None,mode="train"):
        self.transform=transforms.Compose(transforms_)

        filelist_A=os.listdir(root1)
        self.root_A=root1
        self.root_B=root2
        self.files_A=sorted(filelist_A)
        #print("filelist_A:",self.files_A)
        filelist_B=os.listdir(root2)
        self.files_B=sorted(filelist_B)

    def __getitem__(self,index):
        image_A=Image.open(os.path.join(self.root_A,self.files_A[index % len(self.files_A)]))
        image_B=Image.open(os.path.join(self.root_B,self.files_B[index % len(self.files_B)]))

        #Convert grayscale images to rgb
        if image_A.mode!="RGB":
            image_A=to_rgb(image_A)
        if image_B.mode!="RGB":
            image_B=to_rgb(image_B)

        item_A=self.transform(image_A)
        item_B=self.transform(image_B)
        return {"A":item_A,"B":item_B}

    def __len__(self):
        return len(self.files_A)

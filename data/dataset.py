from image_utils import *

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import os

class Full_img_dataset(Data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.bbx_dir = "{0}_bbox".format(img_dir)
        self.bbx_list = os.listdir(self.bbx_dir)

    def __getitem__(self, index):
        bbx_path = os.path.join(self.bbx_dir, self.bbx_list[index])
        img_path = os.path.join(self.img_dir, self.bbx_list[index].split('.')[0] + '.jpg')
        out = out_full_data(img_path)
        return out

    def __len__(self):
        return len(self.bbx_list)

class Instance_img_dataset(Data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.bbx_dir = "{0}_bbox".format(img_dir)
        self.bbx_list = os.listdir(self.bbx_dir)

    def __getitem__(self, index):
        bbx_path = os.path.join(self.bbx_dir, self.bbx_list[index])
        img_path = os.path.join(self.img_dir, self.bbx_list[index].split('.')[0] + '.jpg')
        out = out_instance_data(img_path, bbx_path)
        return out

    def __len__(self):
        return len(self.bbx_list)

class Train_fusion_img_dataset(Data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.bbx_dir = "{0}_bbox".format(img_dir)
        self.bbx_list = os.listdir(self.bbx_dir)

    def __getitem__(self, index):
        bbx_path = os.path.join(self.bbx_dir, self.bbx_list[index])
        img_path = os.path.join(self.img_dir, self.bbx_list[index].split('.')[0] + '.jpg')
        out = out_fusion_train_data(img_path, bbx_path)
        return out

    def __len__(self):
        return len(self.bbx_list)


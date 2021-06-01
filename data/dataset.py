from image_utils import *

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import os

class Full_img_dataset(Data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(self.img_dir)
        self.p1 = 1.0
        self.p2 = 1.0
        self.rgb_thresh = 10
        self.gray_thresh = 0

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        out = out_full_data(img_path)
        return out

    def __len__(self):
        return len(self.img_list)
"""
class Instance_img_dataset():
	def __init__(self, args):


	def __getitem__(self, index):"""

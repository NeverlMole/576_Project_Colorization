import argparse

import numpy as np
import torch
import torch.nn as nn

import tqdm

import os

import sys
sys.path.insert(1, '../models/')
sys.path.insert(2, '../data/')

import model_helper
import data_helper

from image_utils import save_image_from_tensor

img_path = '../image_colored/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='The model to create or load')
    parser.add_argument(
        '-d',
        '--data',
        required=True,
        help='The data to load')
    parser.add_argument(
        '-n',
        '--num',
        type=int,
        help='The number of batches to eval. None means all data.')
    parser.add_argument(
        '-o',
        '--output',
        help='The output folder name.')

    args = parser.parse_args()

    # Load Data
    dataloader = data_helper.get_single_data_loader(args.data,
                                                batch_size=1)

    # Load Model
    model, _ = model_helper.get_model(args.model)

    # Coloring picture
    img_path = img_path + args.model + '_' + args.output + '/'

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    f_loss = nn.SmoothL1Loss(beta=1 / 110.)
    losses = []
    cnt = 0
    for inputs, targets in tqdm.tqdm(dataloader):
        outputs = model(inputs)
        loss = f_loss(outputs, targets)
        losses.append(loss.item())

        inputs = torch.unsqueeze(inputs[0], dim=0)
        outputs = torch.cat((inputs, outputs[0]), dim=0)
        targets = torch.cat((inputs, targets[0]), dim=0)
        t1 = torch.zeros_like(inputs)
        t2 = torch.zeros_like(inputs)
        inputs = torch.cat((inputs, t1, t2), dim=0)

        save_image_from_tensor(img_path + str(cnt) + '_i.jpg', inputs)
        save_image_from_tensor(img_path + str(cnt) + '_o.jpg', outputs)
        save_image_from_tensor(img_path + str(cnt) + '_t.jpg', targets)

        cnt += 1

        if args.num != None and cnt >= args.num:
            break

    print('Total Loss:', np.mean(losses))


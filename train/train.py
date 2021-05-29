import argparse

import numpy as np
import torch
import torch.optim as optim


import sys
sys.path.insert(1, '../models/')
sys.path.insert(2, '../data/')

import model_helper
import data_helper

import normal_runner
import fusion_runner

def train_mode_type(v):
    if v == 'Normal' or v == 'Fusion':
        return v
    else:
        raise argparse.ArgumentTypeError(
                'Train-mode must be either Normal or Fusion')

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
        '-t',
        '--train-mode',
        type=train_mode_type,
        required=True,
        help='The training-mode, either Normal or Fusion')
    parser.add_argument(
        '-e',
        '--epoch',
        type=int,
        required=True,
        help='The number of training epoches')
    parser.add_argument(
        '-s',
        '--save',
        type=bool,
        default=True,
        help='Whether to save the model')

    args = parser.parse_args()

    # Load Data
    trainloader, validloader = data_helper.get_data_loader(args.data)

    # Load Model
    model = model_helper.get_model(args.model)

    # Load runner
    runner = None
    if args.train_mode == 'Normal':
        runner = normal_runner
    elif args.train_mode == 'Fusion':
        runner = fusion_runner

    # Training
    train_losses = []
    valid_losses = []
    learning_rate = 1
    betas = (0.99, 0.999)
    use_cuda = True
    for i in range(args.epoch):  # loop over the dataset multiple times
        optimizer = otorch.optim.Adamax(model.parameters(),
                                        lr=learning_rate,
                                        betas=betas)

        loss = runner.run('train', trainloader, model, optimizer,
                              use_cuda=use_cuda)

        train_losses.append(loss)
        with torch.no_grad():
            loss = runner.run('valid', validloader, model, use_cuda=use_cuda)
            valid_losses.append(loss)

        if args.save:
            model_helper.save_model(model, args.model + '_' + str(i))

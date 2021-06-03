import argparse

import numpy as np
import torch
import torch.optim as optim

import os

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
        type=int,
        default=1,
        help='The epoch intervals to save the model. 0 means not save')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=.001,
        help='The learning rate of the model')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=32,
        help='Batch size')
    parser.add_argument(
        '--no-valid',
        type=bool,
        default=False,
        help='Whether to not test valid set.')
    parser.add_argument(
        '-o',
        '--output-model',
        default=None,
        help='The name of the output model. None means no save.')

    args = parser.parse_args()

    # Load Data
    trainloader, validloader = data_helper.get_data_loader(args.data,
                                                batch_size=args.batch_size)

    # Load Model
    model, _ = model_helper.get_model(args.model)
    model = model.cuda()

    # Load runner
    runner = None
    if args.train_mode == 'Normal':
        runner = normal_runner
    elif args.train_mode == 'Fusion':
        runner = fusion_runner

    # Training
    train_losses = []
    valid_losses = []
    learning_rate = args.learning_rate
    betas = (0.99, 0.999)
    for i in range(args.epoch):  # loop over the dataset multiple times
        optimizer = torch.optim.Adamax(model.parameters(),
                                       lr=learning_rate,
                                       betas=betas)

        loss = runner.run('train', trainloader, model, optimizer)

        train_losses.append(loss)

        if not args.no_valid:
            with torch.no_grad():
                loss = runner.run('valid', validloader, model)
                valid_losses.append(loss)

        if args.output_model != None and args.save > 0 and (i + 1) % args.save == 0:
            model_helper.save_model(model, args.output_model + '_' + str(i))

    if args.output_model != None:
        model_helper.save_model(model, args.output_model)

    log_name = args.output_model + '_' + args.data
    log_path = '../log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = log_path + log_name + '.log'
    with open(log_path, 'w') as f:
        f.write('Train loss:\n')
        for loss in train_losses:
            f.write(str(loss) + '\n')
        f.write('Valid loss:\n')
        for loss in valid_losses:
            f.write(str(loss) + '\n')

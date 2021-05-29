import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np

def run(mode, dataloader, model, optimizer=None, use_cuda = True):
    running_loss = []
    f_loss = nn.SmoothL1Loss()

    actual_labels = []
    predictions = []
    for inputs, t_outputs in tqdm.tqdm(dataloader):
        if use_cuda:
            inputs, t_outputs = inputs.cuda(), t_outputs.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = f_loss(outputs,t_outputs)
        running_loss.append(loss.item())

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss = np.mean(running_loss)

    print(mode, "Loss:", loss)

    return loss

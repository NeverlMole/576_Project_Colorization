import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np

def run(mode, dataloader, model, optimizer=None):
    running_loss = []
    f_loss = nn.SmoothL1Loss()

    actual_labels = []
    predictions = []
    for inputs, targets in tqdm.tqdm(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = model(inputs)
        loss = f_loss(outputs,targets)
        running_loss.append(loss.item())

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss = np.mean(running_loss)

    print(mode, "Loss:", loss)

    return loss

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np

print_every = 10

def run(mode, dataloader, model, optimizer=None):
    running_loss = []
    f_loss = nn.SmoothL1Loss()

    cnt = 0
    for inputs, targets in tqdm.tqdm(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = model(inputs)
        loss = f_loss(outputs,targets)
        running_loss.append(loss.item())

        cnt += 1
        if cnt % 10 == 0:
            print(mode, "Loss:", loss.item())


        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss = np.mean(running_loss)

    return loss

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np

print_every = 10

def run(mode, dataloader, model, optimizer=None, print_every=10, batch_num=None):
    running_loss = []
    cur_loss = []
    f_loss = nn.SmoothL1Loss(beta=1 / 110.)

    cnt = 0
    for inputs, targets in tqdm.tqdm(dataloader):
        if type(inputs) == list:
            # Fusing model input
            inputs = [inputs[0].cuda(), inputs[1].cuda(), inputs[2]]
        else:
            inputs = inputs.cuda()
        targets = targets.cuda()

        outputs = model(inputs)
        loss = f_loss(outputs,targets)
        running_loss.append(loss.item())
        cur_loss.append(loss.item())

        cnt += 1
        if cnt % 10 == 0:
            print(mode, "Loss:", np.mean(cur_loss))
            cur_loss = []

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch_num != None and cnt >= batch_num:
            return

    print(mode, "Loss:", np.mean(running_loss))

    return running_loss

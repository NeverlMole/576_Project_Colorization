import numpy as np
import torch

import dataset

image_path = '../img_data/'

def get_test_data_loader(data_name, batch_size=32):
    '''
    The function should return a tqdm dataloader. The input data_name indicates
    the name of the dataset to return.
    '''
    data_path = '../dataset/' + data_name + '/'
    train_in = torch.Tensor(np.load(data_path + 'train_in.npy'))
    train_tg = torch.Tensor(np.load(data_path + 'train_tg.npy'))
    valid_in = torch.Tensor(np.load(data_path + 'valid_in.npy'))
    valid_tg = torch.Tensor(np.load(data_path + 'valid_tg.npy'))

    train_data = torch.utils.data.TensorDataset(train_in, train_tg)
    valid_data = torch.utils.data.TensorDataset(valid_in, valid_tg)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size)

    return train_loader, valid_loader


def get_data_loader(data_name, batch_size=32):
    train_name = data_name + '/train'
    valid_name = data_name + '/valid'
    train_loader = get_single_data_loader(train_name, batch_size)
    valid_loader = get_single_data_loader(valid_name, batch_size)

    return train_loader, valid_loader


def get_single_data_loader(data_name, batch_size=32):
    if data_name[0] == 'H':
        file_name = data_name[2:]
        data_path = image_path + file_name + '/'
        data_dataset = dataset.Full_img_dataset(img_dir=data_path)

        data_loader = torch.utils.data.DataLoader(data_dataset,
                                                   batch_size=batch_size)

    if data_name[0] == 'I':
        file_name = data_name[2:]
        data_path = image_path + file_name
        data_dataset = dataset.Instance_img_dataset(img_dir=data_path)

        data_loader = torch.utils.data.DataLoader(data_dataset,
                                                   batch_size=batch_size)

    return data_loader


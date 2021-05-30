import numpy as np
import torch

def get_data_loader(data_name, batch_size=32):
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

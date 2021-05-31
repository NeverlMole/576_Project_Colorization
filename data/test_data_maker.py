import numpy as np

import os

train_size = 100
valid_size = 20
graph_size = 256

if __name__ == '__main__':
    dataset_path = '../dataset/'

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    test_dataset_path = dataset_path + 'test/'

    if not os.path.exists(test_dataset_path):
        os.makedirs(test_dataset_path)

    train_in = np.random.rand(train_size, graph_size, graph_size)
    train_tg = np.random.rand(train_size, 2, graph_size, graph_size)
    valid_in = np.random.rand(valid_size, graph_size, graph_size)
    valid_tg = np.random.rand(valid_size, 2, graph_size, graph_size)

    np.save(test_dataset_path + 'train_in', train_in)
    np.save(test_dataset_path + 'train_tg', train_tg)
    np.save(test_dataset_path + 'valid_in', valid_in)
    np.save(test_dataset_path + 'valid_tg', valid_tg)

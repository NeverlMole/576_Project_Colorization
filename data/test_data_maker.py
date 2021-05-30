import numpy as np

import os

train_size = 30
valid_size = 10
graph_size = 256

if __name__ == '__main__':
    dataset_path = '../dataset/'

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    test_dataset_path = dataset_path + 'test/'

    if not os.path.exists(test_dataset_path):
        os.makedirs(test_dataset_path)

    train_in = np.random.rand(train_size, graph_size, graph_size, 1)
    train_tg = np.random.rand(train_size, graph_size, graph_size, 2)
    valid_in = np.random.rand(valid_size, graph_size, graph_size, 1)
    valid_tg = np.random.rand(valid_size, graph_size, graph_size, 2)

    np.save(test_dataset_path + 'train_in', train_in)
    np.save(test_dataset_path + 'train_tg', train_tg)
    np.save(test_dataset_path + 'valid_in', valid_in)
    np.save(test_dataset_path + 'valid_tg', valid_tg)

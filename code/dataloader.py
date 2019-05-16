import os
import pickle
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ACData(Dataset):
    """
    Acient Chinese dataset
    """
    def __init__(self, data_dir, mode='train'):
        """
        mode: train / test
        """
        assert mode == 'train' or mode == 'test'
        self.mode = mode

        if(self.mode == 'train'): 
            with open(os.path.join(data_dir, 'train_origin_ids.pickle'), 'rb') as f:
                self.train_ids = pickle.load(f)
            with open(os.path.join(data_dir, 'train_target_ids.pickle'), 'rb') as f:
                self.target_ids = pickle.load(f)
        else:
            with open(os.path.join(data_dir, 'test_origin_ids.pickle'), 'rb') as f:
                self.test_ids = pickle.load(f)
   
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, item):
        if self.mode == 'train':
            train_x = t.tensor(self.train_ids[item])
            train_y = t.tensor(self.target_ids[item])
            return train_x, train_y
        else:
            test_x = t.tensor(self.test_ids[item])
            return test_x

def ACDataLoader(data_dir, batch_size=1, num_workers=0, shuffle=False):
    datasets = {
        'train_data': ACData(data_dir, 'train'),
        'test_data': ACData(data_dir, 'test')
    }
    dataLoaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       shuffle=shuffle,
                       num_workers=num_workers) for ds in datasets.keys()
    }
    return dataLoaders

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    data_dir = os.path.join(cur_path, '../data/processing')

    dataLoader = ACDataLoader(data_dir)
    for data in dataLoader['train_data']:
        train_x, train_y = data
        print(train_x, train_y)
        break
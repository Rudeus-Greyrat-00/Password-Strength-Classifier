import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class PswDataset(Dataset):
    def __init__(self, dataset, dataset_type='file'):
        if dataset_type == 'file':
            self.psw_label = pd.read_csv(dataset, on_bad_lines='skip')
            self.psw_label.dropna(inplace=True)
        elif dataset_type == 'object':
            self.psw_label = dataset
        else:
            raise ValueError("Allowed values: 'file', 'object'")

    def __len__(self):
        return len(self.psw_label)

    def __getitem__(self, idx):
        word = self.psw_label.iloc[idx, 0]
        label = self.psw_label.iloc[idx, 1]
        return word, label

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.__len__():
            to_return = self.__getitem__(self.n)
            self.n += 1
            return to_return
        raise StopIteration

    def split(self, train_test_split=0.2):
        indexes = np.zeros(len(self))
        indexes[0:int(len(self) * train_test_split)] = 1
        np.random.shuffle(indexes)
        test = self.psw_label[indexes == 1]
        train = self.psw_label[indexes == 0]
        return PswDataset(train, 'object'), PswDataset(test, 'object')


def char_tokenizer(string: str):
    char = []
    for i in string:
        char.append(i)
    return char


def double_char_tokenizer(string: str):
    tokens = []
    for i in range(len(string) - 1):
        tokens.append(string[i:i + 2])
    return tokens

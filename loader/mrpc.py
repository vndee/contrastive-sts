import os
import torch
from torch.utils.data import Dataset, DataLoader


def load():
    file_reader = open(os.path.join('data', 'MRPC', 'dev.tsv'), 'r')
    _file_content = file_reader.read()
    _data = list()

    for line in _file_content.split('\n')[1:]:
        if line.strip() == '':
            continue

        line = line.split('\t')
        _data.append(line)
    file_reader.close()

    return _data


class MRPCDataLoader(Dataset):
    def __init__(self, path=os.path.join('data', 'MRPC', 'dev.tsv')):
        self.path = path
        self.lines = load()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.toList()

        line = self.lines[item]
        

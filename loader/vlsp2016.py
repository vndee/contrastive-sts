import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from loader.tokenizer import PhoBertTokenizer, BertViTokenizer


class VLSP2016(Dataset):
    def __init__(self,
                 file='SA-2016.train',
                 path=os.path.join('data', 'VLSP2016'),
                 max_length=256,
                 tokenizer_type='phobert'):
        super(VLSP2016, self).__init__()
        self.df = pd.read_csv(os.path.join(path, file),
                              names=['sentence', 'label'],
                              sep='\t',
                              encoding='utf-8-sig')

        self.max_length = max_length

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'phobert':
            self.tokenizer = PhoBertTokenizer(max_length=self.max_length)
        else:
            self.tokenizer = BertViTokenizer(max_length=self.max_length, shortcut_pretrained='multilingual-bert-case')

        self.neu = self.df.loc[self.df['label'] == 'NEU']
        self.neg = self.df.loc[self.df['label'] == 'NEG']
        self.pos = self.df.loc[self.df['label'] == 'POS']

        print('Loaded VLSP-2016')
        print(f'There are {len(self.df)} samples in {file} dataset.')

    def __getitem__(self, item):
        neu = self.neu.iloc[item, 0].encode('utf-8').decode('utf-8-sig').strip()
        neg = self.neg.iloc[item, 0].encode('utf-8').decode('utf-8-sig').strip()
        pos = self.pos.iloc[item, 0].encode('utf-8').decode('utf-8-sig').strip()

        neu, neg, pos = self.tokenizer(neu), self.tokenizer(neg), self.tokenizer(pos)
        return torch.cat((neu.unsqueeze(0), neg.unsqueeze(0), pos.unsqueeze(0)))

    def __len__(self):
        return len(self.df)

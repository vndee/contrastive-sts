import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from torch.utils.data import DataLoader, Dataset


DEVICE = 'cuda'
nt_xent_params = {
    'temperature': 0.5,
    'use_cosine_similarity': True
}


class BPEConfig:
    bpe_codes = os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'bpe.codes')


class VNNewsDataset(Dataset):
    def __init__(self, data_dir, max_length=150, remove_negative_pair=True):
        super(VNNewsDataset, self).__init__()
        self.data_dir = data_dir
        self.max_length = max_length

        self.sentence_1 = open(os.path.join(self.data_dir, 'Sentences_1.txt'),
                               mode='r',
                               encoding='utf-8-sig').read().split('\n')

        self.sentence_2 = open(os.path.join(self.data_dir, 'Sentences_2.txt'),
                               mode='r',
                               encoding='utf-8-sig').read().split('\n')

        self.labels = open(os.path.join(self.data_dir, 'Labels.txt'),
                           mode='r',
                           encoding='utf-8-sig').read().split('\n')

        self.bpe = fastBPE(BPEConfig)
        self.vocab = Dictionary()
        self.vocab.add_from_file(os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'dict.txt'))

        if remove_negative_pair is True:
            self.remove_negative_pair()

    def remove_negative_pair(self):
        self.sentence_1 = [sent for idx, sent in enumerate(self.sentence_1) if self.labels[idx] == '1']
        self.sentence_2 = [sent for idx, sent in enumerate(self.sentence_2) if self.labels[idx] == '1']

    def encode(self, raw_text):
        subwords = '<s> ' + self.bpe.encode(raw_text) + ' <s>'
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

        if input_ids.__len__() > self.max_length:
            print(input_ids.__len__())
            raise MemoryError

        input_ids.extend([0] * (self.max_length - input_ids.__len__()))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids

    def __len__(self):
        assert self.sentence_1.__len__() == self.sentence_2.__len__()
        return self.sentence_1.__len__()

    def __getitem__(self, item):
        sent_1 = self.encode(self.sentence_1[item])
        sent_2 = self.encode(self.sentence_2[item])
        lb = self.labels[item]
        return sent_1, sent_2, lb


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, sz):
        self.batch_size = sz
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class PhoBERTCLREncoder(torch.nn.Module):
    def __init__(self):
        super(PhoBERTCLREncoder, self).__init__()

        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.phobert = RobertaModel.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config
        )

        self.linear_1 = torch.nn.Linear(768, 768)
        self.linear_2 = torch.nn.Linear(768, 512)

    def forward(self, inputs, attention_mask):
        phobert_output = self.phobert(inputs,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True)[0]
        features = phobert_output[:, 0, :]
        x = self.linear_1(features)
        x = torch.nn.functional.tanh(x)
        x = self.linear_2(x)
        return x


class PhoBERTCLR(object):
    def __init__(self, batch_size):
        super(PhoBERTCLR, self).__init__()
        self.batch_size = batch_size
        self.phobert_encoder = PhoBERTCLREncoder().to(DEVICE)
        self.nt_xent_criterion = NTXentLoss(DEVICE, batch_size=batch_size, **nt_xent_params)
        self.optimizer = torch.optim.Adam(self.phobert_encoder.parameters(), 3e-10, weight_decay=10-6)

    def train(self, data_loader, num_epochs=20):
        self.phobert_encoder.train()

        for epoch in range(num_epochs):
            avg_loss = 0.0

            for idx, (sentence_1, sentence_2, label) in enumerate(tqdm(data_loader, desc=f'Training EPOCH '
                                                                                         f'[{epoch}/{num_epochs}]')):
                self.optimizer.zero_grad()
                sentence_1 = sentence_1.to(DEVICE)
                sentence_2 = sentence_2.to(DEVICE)

                sent_1 = self.phobert_encoder(sentence_1, (sentence_1 > 0).to(DEVICE))
                sent_2 = self.phobert_encoder(sentence_2, (sentence_1 > 0).to(DEVICE))

                sent_1 = torch.nn.functional.normalize(sent_1, dim=1)
                sent_2 = torch.nn.functional.normalize(sent_2, dim=1)

                loss = self.nt_xent_criterion(sent_1, sent_2, len(label))
                avg_loss = avg_loss + loss.item()

                loss.backward()
                self.optimizer.step()

                # print(f'Epoch [{epoch}/{num_epochs}] - [{idx}/{len(data_loader)}]: {round(loss.item(), 7)}')

            print(f'Epoch [{epoch}/{num_epochs}] done with average loss: {round(avg_loss / len(data_loader), 7)}')


data_dir = os.path.join(os.getcwd(), 'data', 'VNNEWS')
dataset = VNNewsDataset(data_dir, max_length=200, remove_negative_pair=True)
data_loader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=True, drop_last=True)

print(f'Loaded {len(dataset)} samples.')
model = PhoBERTCLR(batch_size=2)
model.train(data_loader, num_epochs=20)

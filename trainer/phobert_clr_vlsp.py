import os
import torch
import numpy as np
from tqdm import tqdm

from vncorenlp import VnCoreNLP
from loader.vlsp2016 import VLSP2016
from models.phobert import PhoBertEncoder
from models.align_uniform import AlignUniformLoss


DEVICE = 'cpu'


class Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.bert = encoder
        self.linear_1 = torch.nn.Linear(768, 256)
        self.linear_2 = torch.nn.Linear(256, 512)

    def forward(self, x, attention_mask):
        x = self.bert(x, attention_mask=attention_mask)[1]
        x = self.linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        return x


class BertCLR(object):
    def __init__(self, encoder):
        super(BertCLR, self).__init__()
        self.net = encoder
        self.encoder = Encoder(self.net)
        self.criterion = AlignUniformLoss(lam=1)

    def train(self, data_loader=None, num_epochs=20, output_dir='outputs'):
        self.encoder.to(DEVICE)

        os.makedirs(output_dir, exist_ok=True)
        optim = torch.optim.SGD(self.encoder.parameters(),
                                lr=0.12*4/256,
                                momentum=0.9,
                                weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         gamma=0.1,
                                                         milestones=[15, 25, 35])

        prev_loss = 9999.9999
        for epoch in range(num_epochs):
            avg_loss, cnt = 0.0, 0
            self.encoder.train()

            for x, item_x in enumerate(tqdm(data_loader, desc=f'EPOCH {epoch}/{num_epochs}')):
                for y in range(x):
                    if x == y:
                        continue

                    item_y = data_loader[y]
                    item_x, item_y = item_x.to(DEVICE), item_y.to(DEVICE)
                    attn_x, attn_y = (item_x > 0).to(DEVICE), (item_y > 0).to(DEVICE)

                    x_z = self.encoder(item_x, attention_mask=attn_x)
                    y_z = self.encoder(item_y, attention_mask=attn_y)

                    loss = self.criterion(x_z, y_z)
                    avg_loss += loss.item()
                    cnt += 1
                    loss.backward()

            print(f'EPOCH {epoch}/{num_epochs} - Loss: {avg_loss/cnt}')
            optim.step()
            scheduler.step()

        
if __name__ == '__main__':
    phobert_enc = PhoBertEncoder()
    clr_net = BertCLR(phobert_enc)

    data_loader = VLSP2016()
    clr_net.train(data_loader)

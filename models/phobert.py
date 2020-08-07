import os
import json
import torch

from transformers import RobertaModel, RobertaConfig, BertPreTrainedModel
from transformers import RobertaForSequenceClassification


class PhoBertEncoder(BertPreTrainedModel):
    def __init__(self, frozen=False, re_init=4):
        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )
        super(PhoBertEncoder, self).__init__(self.config)

        self.phobert = RobertaModel.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config,
        )

        if frozen is True:
            for child in self.phobert.children():
                for param in child.parameters():
                    param.requires_grad = False

        self.init_weights()

    def __call__(self, all_input_ids, attention_mask=None, output_hidden_states=True, output_attentions=None):
        features = self.phobert(all_input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        return features

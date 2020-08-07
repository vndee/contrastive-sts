import torch
from transformers import ElectraTokenizer, ElectraModel


if __name__ == '__main__':
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraModel.from_pretrained('google/electra-small-discriminator')

    input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**input)

    print(outputs[0])

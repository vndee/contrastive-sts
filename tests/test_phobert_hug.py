import torch
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# Load model
config = RobertaConfig.from_pretrained(
    "/Absolute-path-to/PhoBERT_base_transformers/config.json"
)
phobert = RobertaModel.from_pretrained(
    "/Absolute-path-to/PhoBERT_base_transformers/model.bin",
    config=config
)

# Load BPE encoder 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/Absolute-path-to/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("/Absolute-path-to/PhoBERT_base_transformers/dict.txt")

# INPUT TEXT IS WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."  

# Encode the line using fastBPE & Add prefix <s> and suffix </s> 
subwords = '<s> ' + bpe.encode(line) + ' </s>'

# Map subword tokens to corresponding indices in the dictionary
input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

# Convert into torch tensor
all_input_ids = torch.tensor([input_ids], dtype=torch.long)

# Extract features  
with torch.no_grad():  
    features = phobert(all_input_ids)  
  
# Represent each word by the contextualized embedding of its first subword token  
# i. Get indices of the first subword tokens of words in the input sentence 
listSWs = subwords.split()  
firstSWindices = []  
for ind in range(1, len(listSWs) - 1):  
    if not listSWs[ind - 1].endswith("@@"):  
        firstSWindices.append(ind)  

# ii. Extract the corresponding contextualized embeddings  
words = line.split()  
assert len(firstSWindices) == len(words)  
vectorSize = features[0][0, 0, :].size()[0]  
for word, index in zip(words, firstSWindices):  
    print(word + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))
    # print(word + " --> " + listSWs[index] + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))

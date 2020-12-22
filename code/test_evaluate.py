import torchtext
import pandas as pd
from torchtext import datasets
from torchtext import data
from janome.tokenizer import Tokenizer
import janome
import time
import math
import random
import numpy as np
from torchtext.data import Field, BucketIterator
import torch.optim as optim
import torch.nn as nn
import torch

j_t = Tokenizer()

def tokenizer(text):
    return [tok for tok in j_t.tokenize(text, wakati=True)]

SRC = data.Field(sequential=True, tokenize=tokenizer,
                     init_token='<sos>', eos_token='<eos>', lower=True)
TRG = data.Field(sequential=True, tokenize=tokenizer,
                     init_token='<sos>', eos_token='<eos>', lower=True)
train, val, test = data.TabularDataset.splits(
            path="../data/", train='one_train.tsv',
            validation='one_val.tsv', test='one_test.tsv', format='tsv',
            fields=[('SRC', SRC), ('TRG', TRG)])

SRC.build_vocab(train)
TRG.build_vocab(train)

# 各データをバッチ化する
# データの総数を割れる数にしないと学習時にエラーを吐く
train_batch_size = 50
test_batch_size = 32
eval_batch_size = 2
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort=False,
                                                                batch_sizes=(train_batch_size, eval_batch_size, test_batch_size))


for i, batch in enumerate(val_iter):
    print(batch.SRC)
    print(torch.flip(batch.SRC, [0, 1]))
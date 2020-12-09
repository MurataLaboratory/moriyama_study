#!/usr/bin/env python
# coding: utf-8

# [code of transformer from pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)
# 
# 

# In[1]:


print('hello world')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[2]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        # self.decoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        trg_mask = model.generate_square_subsequent_mask(trg.size()[0]).to(device)
        # 分散表現に変換
        src = self.encoder(src)
        trg = self.encoder(trg)
        # 位置情報を入れる
        src = self.pos_encoder(src)
        trg = self.pos_encoder(trg)
        # モデルにデータを入れる
        output = self.transformer_encoder(src)
        # デコーダにエンコーダの出力を入れる（ここがおかしい）
        output = self.transformer_decoder(trg, output,tgt_mask = trg_mask)
        return output


# In[3]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[4]:


get_ipython().system('pip install janome')


# In[5]:


import janome
from janome.tokenizer import Tokenizer
from torchtext import data
from torchtext import datasets
import random
import numpy as np


# In[6]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[23]:


j_t = Tokenizer()
def tokenizer(text): 
    return [tok for tok in j_t.tokenize(text, wakati=True)]


# In[24]:


# 重複のないデータセットか重複のあるデータセットを選ぶ
# flagがTrueの時重複のないデータを返す
def choose_dataset(flag = False):
  if flag:
    train, val, test = data.TabularDataset.splits(
        path="../data/", train='one_train.tsv',
        validation='one_val.tsv', test='one_test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', SRC)])
    filename = "../csv/one_result_transformer.csv"
  else:
    train, val, test = data.TabularDataset.splits(
        path="../data/", train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', SRC)])
    filename = "../csv/result_transformer.csv"
  
  return train, val, test, filename


# In[25]:


SRC = data.Field(sequential=True, 
                 tokenize=tokenizer,
                 init_token='<sos>',
                 eos_token='<eos>', 
                 lower=True)
train, val, test, filename = choose_dataset(True)


# In[26]:


SRC.build_vocab(train, min_freq=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 16
test_batch_size = 32
eval_batch_size = 32
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort = False,  batch_sizes = (train_batch_size,eval_batch_size, test_batch_size), device= device)


# In[27]:


ntokens = len(SRC.vocab.stoi) # the size of vocabulary
emsize = len(SRC.vocab.stoi) # embedding dimension
nhid = 512 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.3 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


# In[ ]:


model


# In[ ]:


criterion = nn.CrossEntropyLoss(ignore_index=SRC.vocab.stoi["<pad>"])
lr = 5 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train(iterator):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for i, batch in enumerate(iterator):
        #print(i)
        src = batch.SRC
        trg = batch.TRG
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[:].view(-1, output.shape[-1])
        trg = trg[:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += float(loss.item())
    
    return total_loss / len(iterator)
        

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
      for i, batch in enumerate(data_source):
        data = batch.SRC
        targets = batch.TRG
        #src_mask = model.generate_square_subsequent_mask(data.shape[0]).to(device)
        output = eval_model(data, targets)
        output_flat = output[:].view(-1, output.shape[-1])
        targets = targets[:].view(-1)
        total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


# In[ ]:


best_val_loss = float("inf")
epochs = 50 # The number of epochs
best_model = None
model.init_weights()
train_loss_list, eval_loss_list = [], []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    t_loss = train(train_iter)
    val_loss = evaluate(model, val_iter)
    print('-' * 89)
    print('| epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          .format(epoch, (time.time() - epoch_start_time), val_loss))
    
    train_loss_list.append(t_loss)
    eval_loss_list.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


# In[ ]:


from matplotlib import pyplot as plt
y = list(range(epochs))
train_loss = plt.plot(y, train_loss_list)
valid_loss = plt.plot(y, eval_loss_list)
plt.xlabel("epochs")
plt.ylabel("train loss")
plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"),)
plt.show()


# In[ ]:


test_loss = evaluate(best_model, test_iter)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# In[ ]:


torch.save(best_model.state_dict(), "/content/dirve/My Drive/Colab Notebooks/model/transformer.pth")


# In[ ]:


model.state_dict(torch.load("/content/dirve/My Drive/Colab Notebooks/model/transformer.pth"))


# In[ ]:


def gen_sentence(sentence, src_field, trg_field, model, batch_size):
  model.eval()
  in_str, out_str, pred, tmp = [], [], [], []
  length = len(sentence)

  with torch.no_grad():
    for _, batch in enumerate(sentence):
      src = batch.SRC
      trg = batch.TRG
      output = model(src, trg)
          
      for j in range(min(length, batch_size)):
        _, topi = output.data.topk(1)
        _, topi_s = output.data.topk(2) 
        for k in range(topi.size()[1]):
          if topi[:, k][0] == trg_field.vocab.stoi["<eos>"]:
            for m in range(topi_s.size()[0]):
              for l in range(topi_s.size()[1]):
                topi[m][l][0] = topi_s[m][l][1]
          for i in range(topi.size()[0]):
            if trg_field.vocab.itos[topi[:, k][i]] == "<eos>":
              break
            tmp.append(trg_field.vocab.itos[topi[:, k][i]])
          pred.append(tmp)
          tmp = []
        #print(src.size())
        in_str.append([src_field.vocab.itos[i.item()] for i in src[:,j] if src_field.vocab.itos[i.item()] != "<eos>"])
        out_str.append([trg_field.vocab.itos[i.item()] for i in trg[:,j] if trg_field.vocab.itos[i.item()] != "<eos>"])
      
  return in_str, out_str, pred


# In[ ]:


# 中間発表時にはテストデータは用いない
test_in, test_out, test_pred = [],[],[]
test_in, test_out, test_pred = gen_sentence(test_iter, SRC, SRC, best_model, test_batch_size)
val_in, val_out, val_pred = [],[],[]
val_in, val_out, val_pred = gen_sentence(val_iter, SRC, SRC, model, eval_batch_size)
train_in, train_out, train_pred = [],[],[]
train_in, train_out, train_pred = gen_sentence(train_iter, SRC, SRC, model, train_batch_size)


# In[ ]:


import pandas as pd


# In[ ]:


def convert_list_to_df(in_list, out_list, pred_list):
  row = []
  for i in range(len(in_list)):
    batch_input = in_list[i]
    batch_output = out_list[i]
    batch_pred = pred_list[i]
    input = [j for j in batch_input if j != "<pad>" and j != "<sos>" and j != "<eos>" and j != "<unk>"]
    output = [j for j in batch_output if j != "<pad>" and j != "<sos>" and j != "<eos>" and j != "<unk>"]
    predict = [j for j in batch_pred if j != "<pad>" and j != "<sos>" and j != "<eos>" and j != "<unk>"]
    input_str = "".join(input)
    output_str ="".join(output)
    predict_str = "".join(predict)
    row.append([input_str, output_str, predict_str])

  df = pd.DataFrame(row, columns=["input","answer","predict"])
  df = df.sort_values('input')
  return df


# In[ ]:


train_df = convert_list_to_df(train_in, train_out, train_pred)
val_df = convert_list_to_df(val_in, val_out, val_pred)
test_df = convert_list_to_df(test_in, test_out, test_pred)


# In[ ]:


df_s = pd.concat([train_df, test_df]).sort_values('input')


# In[ ]:


df_s.head(10)


# In[ ]:


df_s.to_csv(filename)


# In[ ]:





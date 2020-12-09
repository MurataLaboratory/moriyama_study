#!/usr/bin/env python
# coding: utf-8

# # 参考文献
# 
# [Visualizing Bert Embeddings](https://krishansubudhi.github.io/deeplearning/2020/08/27/bert-embeddings-visualization.html)


print('hello world')

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator

import numpy as np

import random
import math
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

import torchtext
import torch
from torchtext import data
from torchtext import datasets
import pandas as pd

from transformers import BertJapaneseTokenizer, BertForPreTraining
tok = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')


def tokenizer(text):
  return tok.tokenize(text)



path = "../data/data.tsv"
src, trg, tmp = [], [], []
with open(path, mode = 'r', encoding="utf-8") as f:
    for file in f:
        sentence = file.split("\t")
        tmp.append(sentence)

random.shuffle(tmp)

for sentence in tmp:
    src.append(sentence[0])
    trg.append(sentence[1].replace("\n", ""))

src_tensors = tok.__call__(text=src, text_pair=trg, padding=True, return_tensors='pt', return_attention_mask=False)
trg_tensors = tok.__call__(text=trg, text_pair=src, padding=True, return_tensors='pt', return_attention_mask=False)


# In[8]:


dataset = torch.utils.data.TensorDataset(src_tensors['input_ids'],
                                        trg_tensors['input_ids'])
train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size
train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])


# In[9]:


batch_size = 32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=True)


# In[10]:


SRC = torchtext.data.Field(sequential=True, tokenize = tokenizer, init_token='<sos>', eos_token='<eos>', lower = True)
TRG = torchtext.data.Field(sequential=True, tokenize = tokenizer, init_token='<sos>', eos_token='<eos>', lower = True)


# In[11]:


# 重複のないデータセットか重複のあるデータセットを選ぶ
# flagがTrueの時重複のないデータを返す
def choose_dataset(flag = False):
  if flag:
    train, val, test = torchtext.data.TabularDataset.splits(
        path="../data/", train='one_train.tsv',
        validation='one_val.tsv', test='one_test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', TRG)])
    filename = "../one_result_bertSeq2seq.csv"
  else:
    train, val, test = torchtext.data.TabularDataset.splits(
        path="../data/", train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', TRG)])
    filename = "../result_bertSeq2seq.csv"
  
  return train, val, test, filename


# In[12]:


train, val, test, filename= choose_dataset(False)


# In[13]:


SRC.build_vocab(train)
TRG.build_vocab(train)


# In[14]:


bert_model = BertForPreTraining.from_pretrained(
    "cl-tohoku/bert-base-japanese", # 日本語Pre trainedモデルの指定
    num_labels = 2, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions = False, # アテンションベクトルを出力するか
    output_hidden_states = True, # 隠れ層を出力するか
)


# In[15]:


class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.hid_dim = hid_dim
    self.n_layers = n_layers
    #self.embedding = nn.Embedding(input_dim, emb_dim)
    self.embedding = bert_model.get_input_embeddings()
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src):
    embedded = self.dropout(self.embedding(src))
    outputs, (hidden, cell) = self.rnn(embedded)
    return hidden, cell


# In[16]:



class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.n_layers = n_layers

    # self.embedding = nn.Embedding(output_dim, emb_dim)
    self.embedding = bert_model.get_input_embeddings()
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

    self.fc_out = nn.Linear(hid_dim, output_dim)
    # self.fc_out = bert_model.get_output_embeddings()
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, input, hidden, cell):
    input = input.unsqueeze(0)
    embedded = self.dropout(self.embedding(input))
    output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
    # print(output.squeeze(0).size())
    prediction = self.fc_out(output.squeeze(0))

    return prediction, hidden, cell


# In[17]:


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    batch_size = trg.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

    hidden, cell = self.encoder(src)

    output = trg[0,:]
    #print(input)

    for t in range(1, trg_len):
      output, hidden, cell = self.decoder(output, hidden, cell)

      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)
      output = (trg[t] if teacher_force else top1)
    
    return outputs


# In[18]:


tok.vocab_size


# In[19]:


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = tok.vocab_size
# OUTPUT_DIM = 3454
ENC_EMB_DIM = 768
DEC_EMB_DIM = 768
ENC_HID_DIM = 512
DEC_HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


# In[20]:


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


# In[21]:


optimizer = optim.Adam(model.parameters())


# In[22]:


criterion = nn.CrossEntropyLoss(ignore_index = 0)


# In[23]:


def train(model, data_loader, optimizer, criterion, clip):
  model.train()

  epoch_loss = 0

  for src, trg in data_loader:
    src = torch.t(src).to(device)
    trg = torch.t(trg).to(device)
    optimizer.zero_grad()

    output = model(src, trg)

    output_dim = output.shape[-1]
    output = output[:].view(-1, output_dim)
    trg = trg[:].contiguous().view(-1)

    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss / len(data_loader)


# In[27]:


def evaluate(model, data_loader, criterion):
  model.eval()

  epoch_loss = 0

  with torch.no_grad():

    for src, trg in data_loader:

        src = torch.t(src).to(device)
        trg = torch.t(trg).to(device)
        output = model(src, trg)

        output_dim = output.shape[-1]

        output = output[:].view(-1, output_dim)
        trg = trg[:].contiguous().view(-1)

        loss = criterion(output, trg)
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


# In[28]:


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins*60))
  return elapsed_mins, elapsed_secs


# In[29]:


epochs = 100
clip = 1
best_valid_loss = float('inf')
best_model = None

for epoch in range(epochs):
    
    start_time = time.time()
    
    train_loss = train(model, train_data_loader, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_data_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = model

    print("-"*65)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')


torch.save(best_model.state_dict(), '../model/bert_embedded_seq2seq.pth')


# In[ ]:


model.state_dict(torch.load("../model/bert_embedded_seq2seq.pth"))


# In[ ]:


def gen_sentence(sentence, tok, model, max_len = 50):
    model.eval()

    sentence = tok.tokenize(sentnece)
    src = [tok.convert_tokens_to_ids("[CLS]")] + tok.convert_tokens_to_ids(sentence) + [tok.convert_tokens_to_ids("[SEP]")]
    
    src = torch.LongTensor([src])
    src = torch.t(src)
    src = src.to(device)

    src_tensor = model.encoder(src)
    src_tensor = model.pos_encoder(src_tensor).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_index = [tok.convert_tokens_to_ids("[CLS]")]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_index[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_index.append(pred_token)
        if pred_token == 3:
            break

        
    if len(trg_tokens) == 2:
        print(trg_tokens)
        trg_tokens = [" "]
        trg_tokens = [src_field.init_token] + trg_tokens + [src_field.eos_token]
        
    predict = tok.convert_ids_to_tokens(trg_index)
    predit = "".join(predict)
    return predict


# In[ ]:


def gen_sentence_list(path): 
  col, pred = [], []
  input, output = [], []
  with open(path, mode = 'r') as f:
    for file_list in f:
      col.append(file_list.split('\t'))
  for i in col:
    input.append(i[0])
    output.append(i[1])

  for sentence in input:
    pred.append(gen_sentence(sentence, SRC, TRG, best_model))
  return input, output, pred


# In[ ]:


path = "/content/dirve/My Drive/Colab Notebooks/data/test.tsv"
test_input, test_output, test_pred = gen_sentence_list(path)


# In[ ]:


path = "/content/dirve/My Drive/Colab Notebooks/data/train.tsv"
train_input, train_output, train_pred = gen_sentence_list(path)


# In[ ]:


path = "/content/dirve/My Drive/Colab Notebooks/data/val.tsv"
val_input, val_output, val_pred = gen_sentence_list(path)


# In[ ]:


import pandas as pd
def convert_list_to_df(input, output, pred_list):
  row = []
  pred = ""
  for i in range(len(input)):
    input_str = input[i]
    output_str = output[i]
    batch_pred = pred_list[i]
    predict = [j for j in batch_pred if j != "<pad>" and j != "<sos>" and j != "<eos>" and j != "<unk>" and j != "[unk]"]
    predict_str = "".join(predict).replace("#", "")
    row.append([input_str, output_str, predict_str])
  df = pd.DataFrame(row, columns=["input","answer","predict"])
  df = df.sort_values('input')
  return df


# In[ ]:


train_df = convert_list_to_df(train_input, train_output, train_pred)
val_df = convert_list_to_df(val_input, val_output, val_pred)
test_df = convert_list_to_df(test_input, test_output, test_pred)


# In[ ]:


df_s = pd.concat([train_df, test_df])
df_s = df_s.sort_values("input")


# In[ ]:


df_s


# In[ ]:


df_s.to_csv(filename)


# In[ ]:


# df_s.to_csv("/content/dirve/My Drive/Colab Notebooks/csv/one_result_bertSeq2seq.csv")


# In[ ]:





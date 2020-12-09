#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('hello world')


# In[2]:


get_ipython().system('pip install torchtext==0.6.0')


# In[3]:


# 必要なモジュールのインポート
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator

import numpy as np

import random
import math
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[5]:


get_ipython().system('pip install janome')


# In[6]:


import janome
from janome.tokenizer import Tokenizer
j_t = Tokenizer()

# 日本語を単語に分割したリストを返す関数
def tokenizer(text):
  return [tok for tok in j_t.tokenize(text, wakati=True)]


# In[7]:


tokenizer("今日は曇りです")


# In[8]:


import torchtext
import torch
from torchtext import data
from torchtext import datasets


# In[9]:


# pytorchのデータフィードの定義（重要！！）
SRC = data.Field(sequential=True, tokenize = tokenizer, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = data.Field(sequential=True, tokenize = tokenizer, init_token='<sos>', eos_token='<eos>', lower=True)


# In[12]:


# 重複のないデータセットか重複のあるデータセットを選ぶ
# flagがTrueの時重複のないデータを返す
def choose_dataset(flag = False):
  if flag:
    train, val, test = data.TabularDataset.splits(
        path="../data/", train='one_train.tsv',
        validation='one_val.tsv', test='one_test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', TRG)])
    filename = "../csv/one_result_Seq2seq.csv"
  else:
    train, val, test = data.TabularDataset.splits(
        path="../data/", train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('SRC', SRC), ('TRG', TRG)])
    filename = "../csv/result_Seq2seq.csv"
  
  return train, val, test, filename


# In[13]:


train, val, test, filename = choose_dataset(False)


# In[14]:


# 辞書の作成
SRC.build_vocab(train)
TRG.build_vocab(train)


# In[15]:


# 各データをバッチ化する
# データの総数を割れる数にしないと学習時にエラーを吐く
train_batch_size = 50
test_batch_size = 32
eval_batch_size = 2
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort = False,  batch_sizes = (train_batch_size,eval_batch_size, test_batch_size), device= device)


# In[16]:


class Encoder(nn.Module):
  # Encoder層の設定
  def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    # 埋め込み層の次元数
    self.hid_dim = hid_dim
    # LSTMレイヤの数
    self.n_layers = n_layers
    # 埋め込み層の作成
    self.embedding = nn.Embedding(input_dim, emb_dim)
    # LSTM層の作成
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src):
    # 単語を分散表現に変換する
    embedded = self.dropout(self.embedding(src))
    # LSTMに単語を入力して、Encoderからの出力とする
    outputs, (hidden, cell) = self.rnn(embedded)
    return hidden, cell


# In[17]:


class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
    super().__init__()
    # 出力の次元数（単語数と同じ）
    self.output_dim = output_dim
    # 隠れ層の数
    self.hid_dim = hid_dim
    # レイヤ数
    self.n_layers = n_layers
    # 埋め込み層
    self.embedding = nn.Embedding(output_dim, emb_dim)
    # LSTM
    self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
    # 出力をするためのやつ
    self.fc_out = nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, input, hidden, cell):
    # 入力を整形する
    input = input.unsqueeze(0)
    # 分散表現に変換
    embedded = self.dropout(self.embedding(input))
    # Decoderからの出力を得る
    output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
    # 正規化する
    prediction = self.fc_out(output.squeeze(0))

    return prediction, hidden, cell


# In[18]:


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

    for t in range(1, trg_len):
      output, hidden, cell = self.decoder(output, hidden, cell)

      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)
      output = (trg[t] if teacher_force else top1)

    return outputs


# In[19]:


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 1024
DEC_HID_DIM = 1024
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


SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = SRC_PAD_IDX)


# In[23]:


def train(model, iterator, optimizer, criterion, clip):
  model.train()

  epoch_loss = 0

  for i, batch in enumerate(iterator):

    src = batch.SRC
    trg = batch.TRG
    optimizer.zero_grad()

    output = model(src, trg)

    #print("output size:", output.size())
    #print("target size:", trg.size())
    output_dim = output.shape[-1]
    output = output[:].view(-1, output_dim)
    trg = trg[:].view(-1)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss / len(iterator)


# In[24]:


def evaluate(model, iterator, criterion):
  model.eval()

  epoch_loss = 0

  with torch.no_grad():

    for i, batch in enumerate(iterator):

      src = batch.SRC
      trg = batch.TRG

      output = model(src, trg)

      output_dim = output.shape[-1]

      output = output[:].view(-1, output_dim)
      trg = trg[:].view(-1)

      loss = criterion(output, trg)
      epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# In[25]:


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins*60))
  return elapsed_mins, elapsed_secs


# In[ ]:


epochs = 100
clip = 1
best_model = None


best_valid_loss = float('inf')
best_model = None
for epoch in range(epochs):
    
    start_time = time.time()
    
    train_loss = train(model, train_iter, optimizer, criterion, clip)
    valid_loss = evaluate(model, val_iter, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = model
        #torch.save(model.state_dict(), 'tut1-model.pt')
    
    print("-"*65)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')


# In[ ]:


torch.save(model.state_dict(), '../model/seq2seq.pth')


# In[ ]:


model.state_dict(torch.load(",./model/seq2seq.pth"))


# In[ ]:


def gen_sentence(sentence, src_field, trg_field, model, max_len = 50):
  model.eval()

  tokens = [src_field.init_token] + tokenizer(sentence) + [src_field.eos_token]
  
  src_index = [src_field.vocab.stoi[i] for i in tokens]
  src_tensor = torch.LongTensor(src_index).unsqueeze(1).to(device)
  src_len = torch.LongTensor([len(src_index)]).to(device)
  with torch.no_grad():
    hidden, cell = model.encoder(src_tensor)
  
  trg_index = [trg_field.vocab.stoi[trg_field.init_token]]
  for i in range(max_len):
    trg_tensor = torch.LongTensor([trg_index[-1]]).to(device)
    with torch.no_grad():
      output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
    
    pred_token = output.argmax(1).item()
    trg_index.append(pred_token)
    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break

  trg_tokens = [trg_field.vocab.itos[i] for i in trg_index]
  if len(trg_tokens) == 2:
    print(trg_tokens)
    trg_tokens = ["-"]
    trg_tokens = [src_field.init_token] + trg_tokens + [src_field.eos_token]
  return trg_tokens


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
    pred.append(gen_sentence(sentence, SRC, TRG, model))
  return input, output, pred


# In[ ]:


path = "./data/test.tsv"
test_input, test_output, test_pred = gen_sentence_list(path)
path = "./data/train.tsv"
train_input, train_output, train_pred = gen_sentence_list(path)
path = "./data/val.tsv"
val_input, val_output, val_pred = gen_sentence_list(path)


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


train_df = convert_list_to_df(train_input, train_output, train_pred)
val_df = convert_list_to_df(val_input, val_output, val_pred)
test_df = convert_list_to_df(test_input, test_output, test_pred)


# In[ ]:


df_s = pd.concat([train_df, test_df]).sort_values('input')


# In[ ]:


df_s.to_csv(filename)


# In[ ]:





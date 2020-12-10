#!/usr/bin/env python
# coding: utf-8

# [code of transformer from pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)

import os


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


import janome
from janome.tokenizer import Tokenizer
from torchtext import data
from torchtext import datasets
import random
import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

j_t = Tokenizer()
def tokenizer(text): 
    return [tok for tok in j_t.tokenize(text, wakati=True)]

# 重複のないデータセットか重複のあるデータセットを選ぶ
# flagがTrueの時重複のないデータを返す
def choose_dataset(flag = False, SRC):
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
    return total_loss / len(data_source)


def gen_sentence(sentence, src_field, trg_field, model, batch_size):
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokens = [src_field.init_token] + tokenizer(sentence) + [src_field.eos_token]
  src = [src_field.vocab.stoi[i] for i in tokens]
  src = torch.LongTensor([src])
  # print(src)
  src = torch.t(src)
  src = src.to(device)

  src_tensor = model.encoder(src)
  src_tensor = model.pos_encoder(src_tensor).to(device)
  src_mask = model.generate_square_subsequent_mask(src_tensor.size()[0]).to(device)
  # print(src_tensor)
  with torch.no_grad():
    src_output = model.transformer_encoder(src_tensor, src_mask)

  trg = trg_field.eos_token
  trg = torch.LongTensor([[trg]]).to(device)
  output = []
  # print("src sizse: ", src_output.size())
  for i in range(max_len):
    # print("trg size: ", trg.size())
    trg_tensor = model.encoder(trg)
    # print(trg_tensor.size())
    trg_tensor = model.pos_encoder(trg_tensor).to(device)
    trg_mask = model.generate_square_subsequent_mask(trg_tensor.size()[0]).to(device)
    with torch.no_grad():
      pred = model.transformer_decoder(trg_tensor, src_output, trg_mask)
    # print("predicit sizes: ", pred.size())
    pred_word_index = pred.argmax(2)[-1]
    # add_word = trg_field.vocab.itos[pred_word_index.item()]
    # print(tok.convert_ids_to_tokens(pred_word_index))
    output.append(pred_word_index)
    if pred_word_index == trg.vocab.stoi[trg_field.eos_token]:
      break

    last_index = torch.LongTensor([[pred_word_index.item()]]).to(device)
    trg = torch.cat((trg, last_index))
    
  # predict = "".join(output)
  predict = [trg_field.vocab.itos[i] for i in output]
  predict = "".join(predict)

  return predict

def gen_sentence_list(path, SRC): 
  col, pred = [], []
  input, output = [], []
  with open(path, mode = 'r', encoding = "utf-8") as f:
    for file_list in f:
      col.append(file_list.split('\t'))
  for i in col:
    input.append(i[0])
    output.append(i[1].replace("\n", ""))

  for sentence in input:
    pred.append(gen_sentence(sentence, SRC, SRC, model))
  return input, output, pred

import pandas as pd


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

df_s = pd.concat([train_df, test_df]).sort_values('input')

df_s.head(10)

df_s.to_csv(filename)

def main():
  os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
  
  print("preparing data...")
  SRC = data.Field(sequential=True, 
                 tokenize=tokenizer,
                 init_token='<sos>',
                 eos_token='<eos>', 
                 lower=True)
  train, val, test, filename = choose_dataset(False, SRC)
  SRC.build_vocab(train, min_freq=1)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_batch_size = 16
  test_batch_size = 32
  eval_batch_size = 32
  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort = False,  batch_sizes = (train_batch_size,eval_batch_size, test_batch_size), device= device)
  
  print("building model...")
  ntokens = len(SRC.vocab.stoi) # the size of vocabulary
  emsize = len(SRC.vocab.stoi) # embedding dimension
  nhid = 512 # the dimension of the feedforward network model in nn.TransformerEncoder
  nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead = 2 # the number of heads in the multiheadattention models
  dropout = 0.3 # the dropout value
  model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
  
  print(model)

  criterion = nn.CrossEntropyLoss(ignore_index=SRC.vocab.stoi["<pad>"])
  lr = 5 # learning rate
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

  
  best_val_loss = float("inf")
  epochs = 50 # The number of epochs
  best_model = None
  model.init_weights()
  train_loss_list, eval_loss_list = [], []

  print("training...")
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


  test_loss = evaluate(best_model, test_iter)
  print('=' * 89)
  print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
      test_loss, math.exp(test_loss)))
  print('=' * 89)
  torch.save(best_model.state_dict(), "../model/transformer.pth")

  model.state_dict(torch.load("../model/transformer.pth"))

  # 中間発表時にはテストデータは用いない
  print("generating sentence from text..")
  path = "../data/test.tsv"
  test_input, test_output, test_pred [], [], []
  test_input, test_output, test_pred = gen_sentence_list(path)
  path = "../data/train.tsv"
  train_input, train_output, train_pred = [], [], []
  train_input, train_output, train_pred = gen_sentence_list(path)
  path = "../data/val.tsv"
  val_input, val_output, val_pred = [], [], []
  val_input, val_output, val_pred = gen_sentence_list(path)

  train_df = convert_list_to_df(train_input, train_output, train_pred)
  val_df = convert_list_to_df(val_input, val_output, val_pred)
  test_df = convert_list_to_df(test_input, test_output, test_pred)

  df_s = pd.concat([train_df, test_df]).sort_values('input')

  print(df_s.head(10))

  df_s.to_csv(filename)
  print("done!")



if __name__ == "__main__":
  main()
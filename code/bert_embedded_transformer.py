#!/usr/bin/env python
# coding: utf-8
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import pandas as pd
from torchtext import data
from torchtext import datasets
from transformers import BertJapaneseTokenizer, BertForPreTraining
import random
import numpy as np
from evaluate import eval_score

bert_model = BertForPreTraining.from_pretrained(
      "cl-tohoku/bert-base-japanese", # 日本語Pre trainedモデルの指定
      num_labels = 2, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
      output_attentions = False, # アテンションベクトルを出力するか
      output_hidden_states = True, # 隠れ層を出力するか
  )

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

# 必要なモジュールのインポート
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=get_freer_gpu())

# device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        # self.linear = nn.Linear(32000 ,768)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_norm = nn.LayerNorm(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      nlayers, norm = encoder_norm)
        self.encoder_embedding = bert_model.get_input_embeddings()
        self.decoder_embedding = bert_model.get_input_embeddings()
        self.ninp = ninp
        decoder_norm = nn.LayerNorm(ninp)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers,
                                                      nlayers
                                                      ,norm=decoder_norm)
        self.linear = bert_model.get_output_embeddings()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        # src_mask = self.generate_square_subsequent_mask(src.size()[0]).to(device)
        trg_mask = self.generate_square_subsequent_mask(trg.size()[0]).to(device)
        # 分散表現に変換
        src = self.encoder_embedding(src)
        # 位置情報を入れる
        src = self.pos_encoder(src)
        # モデルにデータを入れる
        enc_output = self.transformer_encoder(src)
        # enc_output = self.linear(enc_output)
        # print("enc output size: ", enc_output.size())
        # print("trg size: ", trg.size())
        # デコーダにエンコーダの出力を入れる（ここがおかしい）
        trg = self.decoder_embedding(trg)
        trg = self.pos_encoder(trg)
        output = self.transformer_decoder(trg, enc_output, tgt_mask = trg_mask)
        output = self.linear(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.size())
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# https://huggingface.co/transformers/main_classes/tokenizer.html

tok = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenizer(text):
  return tok.tokenize(text)

def cut_src_tensor(tensor):
  for index, padded_tensor in enumerate(tensor):
    # print(padded_tensor.size(0))
    for num, i in enumerate(padded_tensor):
      if i == 3:
        padded_tensor[num] = 0
    sum_pad = sum(0 == padded_tensor).item()
    if sum_pad == padded_tensor.size(0):
      break
  # print(tensor[:index])
  return tensor[:index]

def cut_trg_tensor(tensor):
  for index, padded_tensor in enumerate(tensor):
    # print(padded_tensor.size(0))
    sum_pad = sum(0 == padded_tensor).item()
    if sum_pad == padded_tensor.size(0):
      break
  # print(tensor[:index])
  return tensor[:index]

from tqdm import tqdm

import time
def train(model, data_loader, optimizer, criterion):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for src, trg in data_loader:
        src = torch.t(src).to(device)[1:]
        trg = torch.t(trg).to(device)
        src = cut_src_tensor(src)
        trg = cut_trg_tensor(trg)

        # print("src: ", src)
        # print("trg: ", trg)
        trg_input = trg[:-1]
        optimizer.zero_grad()
        output = model(src, trg_input)
        # print()
        # for index in output.argmax(2):
        #    print(tok.convert_ids_to_tokens(index))
        output = output[:].contiguous().view(-1, output.shape[-1])
        trg = trg[1:].contiguous().view(-1)
        # print("trg: ", trg)
        # print("trg size :", trg.size())
        # print("output size: ", output.size())
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item() * len(src)

    return total_loss / len(data_loader)


def evaluate(eval_model, data_loader, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
      for src, trg in data_loader:
        src = torch.t(src).to(device)[1:]
        trg = torch.t(trg).to(device)
        src = cut_src_tensor(src)
        trg = cut_trg_tensor(trg)
        #src_mask = model.generate_square_subsequent_mask(data.shape[0]).to(device)
        trg_input = trg[:-1]
        output = eval_model(src, trg_input)
        output_flat = output[:].contiguous().view(-1, output.shape[-1])
        trg = trg[1:].contiguous().view(-1)
        total_loss += criterion(output_flat, trg).item() * len(src)
    return total_loss / (len(data_loader) - 1)

def gen_sentence(sentence, tok, model, max_len = 50):
  model.eval()

  sentence = tok.tokenize(sentence)
  # src = [tok.convert_tokens_to_ids("[CLS]")] + tok.convert_tokens_to_ids(sentence) + [tok.convert_tokens_to_ids("[SEP]")]
  src = tok.convert_tokens_to_ids(sentence)#  + [tok.convert_tokens_to_ids("[SEP]")]
  src = torch.LongTensor([src])
  src = torch.t(src)
  src = src.to(device)
  # print(src)
  trg = tok.convert_tokens_to_ids("[CLS]")
  trg = torch.LongTensor([[trg]]).to(device)
  output = []
  # print("src sizse: ", src_output.size())
  for i in range(max_len):
    with torch.no_grad():
      pred = model(src, trg)
    # print("predicit sizes: ", pred.size())
    pred_word_index = pred.argmax(2)[-1]
    # add_word = trg_field.vocab.itos[pred_word_index.item()]
    # print(tok.convert_ids_to_tokens(pred_word_index))
    if pred_word_index == 3:
      break

    output.append(pred_word_index)

    last_index = torch.LongTensor([[pred_word_index.item()]]).to(device)
    trg = torch.cat((trg, last_index))
  # predict = "".join(output)
  predict = tok.convert_ids_to_tokens(output)
  predict = "".join(predict)
  # print(predict)
  return predict


def gen_sentence_list(model, path):
  col, pred = [], []
  input, output = [], []
  with open(path, mode = 'r', encoding = "utf-8") as f:
    for file_list in f:
      col.append(file_list.split('\t'))
  for i in col:
    input.append(i[0])
    output.append(i[1].replace("\n", ""))

  bar = tqdm(total = len(input))
  for sentence in input:
    pred.append(gen_sentence(sentence, tok, model))
    bar.update(1)
  return input, output, pred

def convert_list_to_df(in_list, out_list, pred_list):
  row = []
  for i in range(len(in_list)):
    batch_input = in_list[i]
    batch_output = out_list[i]
    batch_pred = pred_list[i]
    input = [j for j in batch_input if j != "<pad>" and j != "<sos>" and j != "[SEP]" and j != "<unk>"]
    output = [j for j in batch_output if j != "<pad>" and j != "<sos>" and j != "[SEP]" and j != "<unk>"]
    predict = [j for j in batch_pred if j != "<pad>" and j != "<sos>" and j != "[SEP]" and j != "<unk>"]
    input_str = "".join(input).replace("#", "")
    output_str ="".join(output).replace("#", "")
    predict_str = "".join(predict).replace("#", "")
    row.append([input_str, output_str, predict_str])

  df = pd.DataFrame(row, columns=["input","answer","predict"])
  df = df.sort_values('input')
  return df

def main():
  os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

  SEED = 1234

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True


  print("preparing data..")
  paths = ["../data/train.tsv", "../data/val.tsv"]
  src, trg, tmp = [], [], []
  for path in paths:
    with open(path, mode='r', encoding = "utf-8") as f:
      for file in f:
        sentence = file.split("\t")
        tmp.append(sentence)

  # random.shuffle(tmp)
  for sentence in tmp:
      src.append(sentence[0])
      trg.append(sentence[1].replace("\t", ""))

  src_tensors = tok(text = src, padding=True, return_tensors='pt', return_attention_mask=False)
  trg_tensors = tok(text = trg, padding=True, return_tensors='pt', return_attention_mask=False)

  dataset = torch.utils.data.TensorDataset(src_tensors['input_ids'], trg_tensors['input_ids'])

  train_size = int(len(dataset) * 0.8)
  valid_size = len(dataset) - train_size
  train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])

  batch_size = 64
  # batch_size = 8
  train_data_loader = torch.utils.data.DataLoader(train_data, batch_size)
  valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size)

  print("building model...")
  emsize = 768 # embedding dimension
  nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
  nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead = 2 # the number of heads in the multiheadattention models
  dropout = 0.3 # the dropout value
  model = TransformerModel(emsize, nhead, nhid, nlayers, dropout).to(device)

  print(model)
  criterion = nn.CrossEntropyLoss(ignore_index=tok.convert_tokens_to_ids("[UNK]"))
  lr = 0.0001 # learning rate
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


  best_val_loss = float("inf")
  epochs = 100 # The number of epochs
  best_model = None
  model.init_weights()
  train_loss_list, eval_loss_list = [], []

  print("training model...")
  for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      t_loss = train(model, train_data_loader, optimizer, criterion)
      val_loss = evaluate(model, valid_data_loader, criterion)
      print('-' * 89)
      print('| epoch {:3d} | time: {:3d}m {:3d}s | train loss {:5.2f} | valid loss {:5.2f} | '
              .format(epoch, int((time.time() - epoch_start_time)/60), int((time.time() - epoch_start_time)%60), t_loss, val_loss))

      train_loss_list.append(t_loss)
      eval_loss_list.append(val_loss)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
      model.eval()
      sentence = "今日は良い日ですね"
      sentence = tok.tokenize(sentence)
      # src = [tok.convert_tokens_to_ids("[CLS]")] + tok.convert_tokens_to_ids(sentence) + [tok.convert_tokens_to_ids("[SEP]")]
      src = tok.convert_tokens_to_ids(sentence)# + [tok.convert_tokens_to_ids("[SEP]")]
      src = torch.LongTensor([src])
      src = torch.t(src)
      src = src.to(device)
      trg = tok.convert_tokens_to_ids("[CLS]")
      trg = torch.LongTensor([[trg]]).to(device)
      output = []
      for i in range(25):
          with torch.no_grad():
            pred = model(src, trg)
          pred_word_index = pred.argmax(2)[-1]
          output.append(pred_word_index)
          if pred_word_index == 3:
            break

          last_index = torch.LongTensor([[pred_word_index.item()]]).to(device)
          trg = torch.cat((trg, last_index))
      predict = tok.convert_ids_to_tokens(output)
      print("source sentence: ", sentence)
      print("predicted sentence: ", predict)
      scheduler.step()

  torch.save(best_model.state_dict(), "../model/bert_embedded_transformer.pth")

  # model.init_weights()

  # model.state_dict(torch.load("../model/bert_embedded_transformer.pth"))

  print("generating sentence from text..")
  path = "../data/test.tsv"
  test_input, test_output, test_pred = [], [], []
  test_input, test_output, test_pred = gen_sentence_list(model, path)
  path = "../data/train.tsv"
  train_input, train_output, train_pred = [], [], []
  train_input, train_output, train_pred = gen_sentence_list(model, path)
  path = "../data/val.tsv"
  val_input, val_output, val_pred = [], [], []
  val_input, val_output, val_pred = gen_sentence_list(model, path)

  print("converting list to dataframe")
  train_df = convert_list_to_df(train_input, train_output, train_pred)
  val_df = convert_list_to_df(val_input, val_output, val_pred)
  test_df = convert_list_to_df(test_input, test_output, test_pred)

  df_s = pd.concat([train_df, test_df]).sort_values('input').reset_index().drop(columns = ["index"])

  print(df_s.head(10))

  df_s.to_csv("../csv/result_bert_embedded_transformer.csv")

  df_result = df_s.groupby(["input", "predict"], as_index=False).agg({
        "answer": list
    })

  percentage, kinds, bleu = eval_score(df_result)
  print(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")

  with open("score/score_bert_embedded_transformer.txt", mode="w") as f:
        f.write(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
  print("done!")

if __name__ == "__main__":
  main()



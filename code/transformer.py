#!/usr/bin/env python
# coding: utf-8

# [code of transformer from pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)

import pandas as pd
import time
import numpy as np
import random
from torchtext import datasets
from torchtext import data
from janome.tokenizer import Tokenizer
import janome
import os
from evaluate import eval_score


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

# 必要なモジュールのインポート
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=get_freer_gpu())
# device = torch.device("cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        if ninp%2 == 1:
          ninp += 1
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
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        src_mask = self.generate_square_subsequent_mask(
            src.size()[0]).to(device)
        trg_mask = self.generate_square_subsequent_mask(
            trg.size()[0]).to(device)
        # 分散表現に変換
        src = self.encoder(src)
        trg = self.encoder(trg)
        # 位置情報を入れる
        src = self.pos_encoder(src)
        trg = self.pos_encoder(trg)
        # モデルにデータを入れる
        output = self.transformer_encoder(src, mask = src_mask)
        # デコーダにエンコーダの出力を入れる（ここがおかしい）
        output = self.transformer_decoder(trg, output, tgt_mask=trg_mask)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #if d_model%2 != 0:
        #  d_model += 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #print(pe[:, 0::2].size())
        #print(pe[:, 1::2].size())
        #print((position*div_term).size())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


def choose_dataset(flag, SRC):
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

from tqdm import tqdm

def train_model(model, iterator, optimizer, criterion):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for _, batch in enumerate(iterator):
        # print(i)
        src = batch.SRC
        trg = batch.TRG
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[:].contiguous().view(-1, output.shape[-1])
        # print(output)
        trg = trg[:].contiguous().view(-1)
        # print(trg)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += float(loss.item()) * len(src)

    return total_loss / len(iterator)


def evaluate_model(eval_model, data_source, criterion):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data = batch.SRC
            targets = batch.TRG
            #src_mask = model.generate_square_subsequent_mask(data.shape[0]).to(device)
            output = eval_model(data, targets)
            output_flat = output[:].contiguous().view(-1, output.shape[-1])
            targets = targets[:].contiguous().view(-1)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_source)


def gen_sentence(sentence, src_field, trg_field, model, max_len = 50):
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = [src_field.init_token] + \
        tokenizer(sentence) + [src_field.eos_token]
    src = [src_field.vocab.stoi[i] for i in tokens]
    src = torch.LongTensor([src])
    # print(src)
    src = torch.t(src)
    src = src.to(device)

    src_tensor = model.encoder(src)
    src_tensor = model.pos_encoder(src_tensor).to(device)
    # src_mask = model.generate_square_subsequent_mask(src_tensor.size()[0]).to(device)
    # print(src_tensor)
    with torch.no_grad():
        src_output = model.transformer_encoder(src_tensor)

    trg = trg_field.vocab.stoi[trg_field.eos_token]
    trg = torch.LongTensor([[trg]]).to(device)
    output = []
    # print("src sizse: ", src_output.size())
    for i in range(max_len):
        # print("trg size: ", trg.size())
        # print([trg_field.vocab.itos[i] for i in trg])
        trg_tensor = model.encoder(trg)
        # print(trg_tensor.size())
        trg_tensor = model.pos_encoder(trg_tensor).to(device)
        # trg_mask = model.generate_square_subsequent_mask(trg_tensor.size()[0]).to(device)
        with torch.no_grad():
            pred = model.transformer_decoder(trg_tensor, src_output)
        # print("predicit sizes: ", pred.size())
        pred_word_index = pred.argmax(2)[-1]
        # add_word = trg_field.vocab.itos[pred_word_index.item()]
        output.append(pred_word_index)
        if pred_word_index == trg_field.vocab.stoi[trg_field.eos_token]:
            break

        last_index = torch.LongTensor([[pred_word_index.item()]]).to(device)
        trg = torch.cat((trg, last_index))

    # predict = "".join(output)
    predict = [trg_field.vocab.itos[i] for i in output]
    predict = "".join(predict)

    return predict


def gen_sentence_list(model, path, SRC):
    col, pred = [], []
    input, output = [], []
    with open(path, mode='r', encoding="utf-8") as f:
        for file_list in f:
            col.append(file_list.split('\t'))
    for i in col:
        input.append(i[0])
        output.append(i[1].replace("\n", ""))
    bar = tqdm(total=len(input))
    for sentence in input:
        pred.append(gen_sentence(sentence, SRC, SRC, model))
        bar.update(1)
    return input, output, pred


def convert_list_to_df(in_list, out_list, pred_list):
    row = []
    for i in range(len(in_list)):
        batch_input = in_list[i]
        batch_output = out_list[i]
        batch_pred = pred_list[i]
        input = [j for j in batch_input if j != "<pad>" and j !=
                 "<sos>" and j != "<eos>" and j != "<unk>"]
        output = [j for j in batch_output if j != "<pad>" and j !=
                  "<sos>" and j != "<eos>" and j != "<unk>"]
        predict = [j for j in batch_pred if j != "<pad>" and j !=
                   "<sos>" and j != "<eos>" and j != "<unk>"]
        input_str = "".join(input)
        output_str = "".join(output)
        predict_str = "".join(predict)
        row.append([input_str, output_str, predict_str])

    df = pd.DataFrame(row, columns=["input", "answer", "predict"])
    df = df.sort_values('input')
    return df


def main():
    print("preparing data...")
    SRC = data.Field(sequential=True,
                     tokenize=tokenizer,
                     init_token='<sos>',
                     eos_token='<eos>',
                     lower=True)
    train, val, test, filename = choose_dataset(False, SRC)
    SRC.build_vocab(train, min_freq=1)

    train_batch_size = 16
    test_batch_size = 32
    eval_batch_size = 32
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort=False,  batch_sizes=(
        train_batch_size, eval_batch_size, test_batch_size), device=device)

    print("building model...")
    ntokens = len(SRC.vocab.stoi)  # the size of vocabulary
    emsize = len(SRC.vocab.stoi)  # embedding dimension
    nhid = 1024  # the dimension of the feedforward network model in nn.TransformerEncoder and nn.TransformerDecoder
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder and nn.TransformerDecoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.3  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead,
                             nhid, nlayers, dropout).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss(ignore_index=SRC.vocab.stoi["<pad>"])
    lr = 5  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 100  # The number of epochs
    best_model = None
    model.init_weights()

    print("training...")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        t_loss = train_model(model, train_iter, optimizer, criterion)
        val_loss = evaluate_model(model, val_iter, criterion)
        print('-' * 89)
        print('| epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | '
              .format(epoch, (time.time() - epoch_start_time), t_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    test_loss = evaluate_model(best_model, test_iter, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    torch.save(best_model.state_dict(), "../model/transformer.pth")

    model.state_dict(torch.load("../model/transformer.pth", map_location=device))

    # 中間発表時にはテストデータは用いない
    print("generating sentence from text..")
    path = "../data/test.tsv"
    test_input, test_output, test_pred = [], [], []
    test_input, test_output, test_pred = gen_sentence_list(model, path, SRC)

    path = "../data/train.tsv"
    train_input, train_output, train_pred = [], [], []
    train_input, train_output, train_pred = gen_sentence_list(model, path, SRC)
    path = "../data/val.tsv"
    val_input, val_output, val_pred = [], [], []
    val_input, val_output, val_pred = gen_sentence_list(model, path, SRC)

    train_df = convert_list_to_df(train_input, train_output, train_pred)
    val_df = convert_list_to_df(val_input, val_output, val_pred)
    test_df = convert_list_to_df(test_input, test_output, test_pred)

    df_s = pd.concat([train_df, test_df]).sort_values('input')

    df_s.to_csv(filename)

    df_result = df_s.groupby(["input", "predict"], as_index=False).agg({
        "answer": list
    })

    percentage, kinds, bleu = eval_score(df_result)
    print(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
    with open("score_transformer.txt", mode="w") as f:
        f.write(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
    print("done!")


if __name__ == "__main__":
    main()

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=get_freer_gpu())

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):

    def __init__(self, in_token, out_token, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_norm = nn.LayerNorm(ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)
        self.encoder_embedding = nn.Embedding(in_token, ninp)
        self.decoder_embedding = nn.Embedding(out_token, ninp)
        self.ninp = ninp
        decoder_norm = nn.LayerNorm(ninp)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, norm = decoder_norm)
        self.linear = nn.Linear(ninp, out_token)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        # src_mask = model.generate_square_subsequent_mask(src.size()[0]).to(device)
        trg_mask = self.generate_square_subsequent_mask(trg.size()[0]).to(device)
        # 分散表現に変換
        src = self.encoder_embedding(src)
        # 位置情報を入れる
        src = self.pos_encoder(src)
        # モデルにデータを入れる
        output = self.transformer_encoder(src) #, mask = src_mask)
        # デコーダにエンコーダの出力を入れる
        # print("output size:", output.size())
        # print("trg size: ", trg.size())
        trg = self.decoder_embedding(trg)
        trg = self.pos_encoder(trg)
        output = self.transformer_decoder(trg, output,tgt_mask = trg_mask)
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
        #print(pe[:, 0::2].size())
        #print(pe[:, 1::2].size())
        #print((position*div_term).size())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
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


def choose_dataset(flag, SRC, TRG):
    if flag:
        train, val, test = data.TabularDataset.splits(
            path="../data/", train='one_train.tsv',
            validation='one_val.tsv', test='one_test.tsv', format='tsv',
            fields=[('SRC', SRC), ('TRG', TRG)])
        filename = "../csv/one_result_transformer.csv"
    else:
        train, val, test = data.TabularDataset.splits(
            path="../data/", train='train.tsv',
            validation='val.tsv', test='test.tsv', format='tsv',
            fields=[('SRC', SRC), ('TRG', TRG)])
        filename = "../csv/result_transformer.csv"

    return train, val, test, filename

from tqdm import tqdm

def train_model(model, iterator, optimizer, criterion, SRC):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for _, batch in enumerate(iterator):
        src = batch.SRC
        trg = batch.TRG
        trg_input = trg[:-1]
        # print(src)
        optimizer.zero_grad()
        output = model(src, trg_input)
        # print(output.argmax(2))
        output = output[:].contiguous().view(-1, output.shape[-1])
        trg = trg[1:].contiguous().view(-1)
        # print("trg size: ", trg.size())
        loss = criterion(output, trg)
        #print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item() * len(src)

    return total_loss / len(iterator)


def evaluate_model(eval_model, data_source, criterion):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data = batch.SRC
            targets = batch.TRG
            targets_input = targets[:-1]
            #src_mask = model.generate_square_subsequent_mask(data.shape[0]).to(device)
            output = eval_model(data, targets_input)
            output_flat = output[:].contiguous().view(-1, output.shape[-1])
            targets = targets[1:].contiguous().view(-1)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_source)


def gen_sentence(sentence, src_field, trg_field, model, max_len = 50):
    model.eval()

    tokens = [src_field.init_token] + \
        tokenizer(sentence) + [src_field.eos_token]
    tokens = tokenizer(sentence)
    src = [src_field.vocab.stoi[i] for i in tokens]
    src = torch.LongTensor([src])
    src = torch.t(src)
    src = src.to(device)
    trg = trg_field.vocab.stoi[trg_field.init_token]
    trg = torch.LongTensor([[trg]]).to(device)
    output = []
    # print("src sizse: ", src_output.size())
    for i in range(max_len):
        with torch.no_grad():
            # pred = model.transformer_decoder(trg_tensor, src_output, tgt_mask = trg_mask)
            pred = model(src, trg)
        pred_word_index = pred.argmax(2)[-1]
        # add_word = trg_field.vocab.itos[pred_word_index.item()]
        if pred_word_index == trg_field.vocab.stoi[trg_field.eos_token]:
            break
        output.append(pred_word_index)

        last_index = torch.LongTensor([[pred_word_index.item()]]).to(device)
        trg = torch.cat((trg, last_index))

    # predict = "".join(output)
    predict = [trg_field.vocab.itos[i] for i in output]
    predict = "".join(predict)
    # print(predict)

    return predict


def gen_sentence_list(model, path, SRC, TRG):
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
        pred.append(gen_sentence(sentence, SRC, TRG, model))
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


def prepare_df(df):
    ans_df = pd.read_table("../data/data.tsv", header = None, names = ["input", "answer"]).sort_values('input')
    eval_df = pd.DataFrame(index=[], columns=["input", "answer"])
    ans_df = ans_df.groupby(["input"], as_index = False).agg({
        "answer" : list
    })
    df = df.groupby(["input", "predict"], as_index = False).agg({
        "answer" : list
    })
    df = df.sort_values("input")
    # print(df)
    for _, input_str in enumerate(df["input"]):
        eval_df = eval_df.append(ans_df[ans_df["input"] == input_str])

    eval_df = eval_df.sort_values("input").reset_index(drop = True)
    eval_df["predict"] = df["predict"]
    # print(eval_df)
    return eval_df

def main():
    print("preparing data...")
    SRC = data.Field(tokenize=tokenizer,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)
    TRG = data.Field(tokenize=tokenizer,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)
    train, val, test, filename = choose_dataset(False, SRC, TRG)
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    train_batch_size = 128
    test_batch_size = 32
    eval_batch_size = 128
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort=False,  batch_sizes=(
        train_batch_size, eval_batch_size, test_batch_size), device=device)

    print("building model...")
    in_tokens = len(SRC.vocab.stoi)  # the size of vocabulary
    out_tokens = len(TRG.vocab.stoi)
    emsize = 768 # embedding dimension
    nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder and nn.TransformerDecoder
    nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder and nn.TransformerDecoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.3  # the dropout value
    model = TransformerModel(in_tokens, out_tokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi["<unk>"])
    lr = 0.0001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 100  # The number of epochs
    best_model = None
    # model.init_weights()

    print("training...")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        t_loss = train_model(model, train_iter, optimizer, criterion, SRC)
        val_loss = evaluate_model(model, val_iter, criterion)
        print('-' * 65)
        print('| epoch {:3d} | time: {:3d}m {:3d}s | train loss {:5.2f} | valid loss {:5.2f}'
              .format(epoch, int((time.time() - epoch_start_time)/60), int((time.time() - epoch_start_time)%60), t_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
        model.eval()
        sentence = "今日はいい日ですね"
        output = []
        sentence = SRC.preprocess(sentence)
        # print(sentence)
        index = [SRC.vocab.stoi[SRC.init_token]] + [SRC.vocab.stoi[i] for i in sentence] + [SRC.vocab.stoi[SRC.eos_token]]
        src_tensor = torch.LongTensor([index]).T.to(device)
        trg = torch.LongTensor([[TRG.vocab.stoi[TRG.init_token]]]).to(device)
        for i in range(25):
            pred = model(src_tensor, trg)

            pred_index = pred.argmax(2)[-1].item()
            # print(pred_index)
            output.append(pred_index)
            if pred_index == TRG.vocab.stoi[TRG.eos_token]:
                break

            pred_index = torch.LongTensor([[pred_index]]).to(device)
            # print(pred_index.size())
            trg = torch.cat((trg, pred_index))

        print("source sentence: ", sentence)
        print("output sentence: ", [TRG.vocab.itos[i] for i in output])

    test_loss = evaluate_model(best_model, test_iter, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    torch.save(model.state_dict(), "../model/transformer.pth")

    model.state_dict(torch.load("../model/transformer.pth", map_location=device))
    # print(model.state_dict())
    # 中間発表時にはvalidデータは用いない
    print("generating sentence from text..")
    path = "../data/test.tsv"
    test_input, test_output, test_pred = [], [], []
    test_input, test_output, test_pred = gen_sentence_list(model, path, SRC, TRG)
    path = "../data/train.tsv"
    train_input, train_output, train_pred = [], [], []
    train_input, train_output, train_pred = gen_sentence_list(model, path, SRC, TRG)

    train_df = convert_list_to_df(train_input, train_output, train_pred)
    test_df = convert_list_to_df(test_input, test_output, test_pred)


    test_df = prepare_df(test_df)
    test_percentage, test_kinds, test_bleu = eval_score(test_df)
    train_df = prepare_df(train_df)
    train_percentage, train_kinds, train_bleu = eval_score(train_df)
    train_df.to_csv("../csv/train/result_transformer.csv")
    test_df.to_csv("../csv/test/result_transformer.csv")
    print(f"TEST DATA: 一致率: {test_percentage}, 種類数: {test_kinds}, BLEU: {test_bleu}")
    print(f"TRAIN DATA: 一致率: {train_percentage}, 種類数: {train_kinds}, BLEU: {train_bleu}")
    with open("./score/score_transformer.txt", mode="w") as f:
        f.write(f"TEST DATA: 一致率: {test_percentage}, 種類数: {test_kinds}, BLEU: {test_bleu}\n")
        f.write(f"TRAIN DATA: 一致率: {train_percentage}, 種類数: {train_kinds}, BLEU: {train_bleu}")
    print("done!")

if __name__ == "__main__":
    main()

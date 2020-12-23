#!/usr/bin/env python
# coding: utf-8

# # 参考文献
#
# [Visualizing Bert Embeddings](https://krishansubudhi.github.io/deeplearning/2020/08/27/bert-embeddings-visualization.html)


from tqdm import tqdm
from transformers import BertJapaneseTokenizer, BertForPreTraining
import pandas as pd
from torchtext import datasets
from torchtext import data
import torchtext
import time
import math
import random
import numpy as np
from torchtext.data import Field, BucketIterator
import torch.optim as optim
import torch.nn as nn
import torch
import os
from evaluate import eval_score
print('hello world')


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


# 必要なモジュールのインポート
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu", index=get_freer_gpu())
# device = torch.device("cpu")

bert_model = BertForPreTraining.from_pretrained(
    "cl-tohoku/bert-base-japanese",  # 日本語Pre trainedモデルの指定
    num_labels=2,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions=False,  # アテンションベクトルを出力するか
    output_hidden_states=True,  # 隠れ層を出力するか
)
tok = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')


def tokenizer(text):
    return tok.tokenize(text)


# 重複のないデータセットか重複のあるデータセットを選ぶ
# flagがTrueの時重複のないデータを返す
def choose_dataset(flag, SRC, TRG):
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


class Encoder(nn.Module):
    def __init__(self,  emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = bert_model.get_input_embeddings()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding = bert_model.get_input_embeddings()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)
        # self.fc_out = bert_model.get_output_embeddings()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # print(type(input))
        input = input.unsqueeze(0)
        # print(input.size())
        embedded = self.dropout(self.embedding(input))
        # print(embedded.size())
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print(output.squeeze(1).size())
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell

    def gen_prediction(self, input, hidden, cell):
        # print(type(input))
        # input = input.unsqueeze(0)
        # print(input.size())
        embedded = self.embedding(input)
        # print(embedded.size())
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print(output.squeeze(1).size())
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        output = trg[0, :]
        # print(input)

        for t in range(1, trg_len):
            # output = output.unsqueeze(0)
            # print(output)
            # print(output.size())
            output, hidden, cell = self.decoder(output, hidden, cell)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = (trg[t] if teacher_force else top1)
            # print(output)

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(model, data_loader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for src, trg in data_loader:
        src = torch.t(src).to(device)
        trg = torch.t(trg).to(device)
        src = src.to('cpu').detach().numpy().copy()
        src = np.flipud(src)
        src = torch.from_numpy(src.astype(np.int32)).clone()
        src = src.long().to(device)
        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for src, trg in data_loader:

            src = torch.t(src).to(device)
            trg = torch.t(trg).to(device)
            src = src.to('cpu').detach().numpy().copy()
            src = np.flipud(src)
            src = torch.from_numpy(src.astype(np.int32)).clone()
            src = src.long().to(device)
            output = model(src, trg)

            output_dim = output.shape[-1]

            output = output[:].view(-1, output_dim)
            trg = trg[:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    return elapsed_mins, elapsed_secs


def gen_sentence(sentence, tok, model, max_len=50):
    model.eval()

    sentence = tok.tokenize(sentence)
    src = [tok.convert_tokens_to_ids(
        "[CLS]")] + tok.convert_tokens_to_ids(sentence) + [tok.convert_tokens_to_ids("[SEP]")]

    src = torch.LongTensor([src])
    src = torch.t(src)
    src_tensor = src.to(device)
    # print(src)
    # print(src.size())
    # src_tensor = model.encoder(src)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_index = [tok.convert_tokens_to_ids("[CLS]")]
    for i in range(max_len):
        # print(trg_index)
        trg_tensor = torch.LongTensor([trg_index]).to(device)
        trg_tensor = torch.t(trg_tensor)
        # print(trg_tensor.size())
        # print(trg_tensor)
        with torch.no_grad():
            output, hidden, cell = model.decoder.gen_prediction(
                trg_tensor, hidden, cell)

        # print(output.size())
        # print(output)
        # print(output[-1].argmax(0).item())
        pred_token = output[-1].argmax(0).item()
        trg_index.append(pred_token)
        if pred_token == tok.convert_tokens_to_ids("[SEP]"):
            break

    if len(trg_index) == 2:
        # print(trg_index)
        trg_index = ["-"]
        trg_index = [tok.convert_tokens_to_ids(
            "[CLS]")] + trg_index + [tok.convert_tokens_to_ids("[SEP]")]

    predict = tok.convert_ids_to_tokens(trg_index)
    predit = "".join(predict)
    return predict


def gen_sentence_list(model, path, tok):
    col, pred = [], []
    input, output = [], []
    with open(path, mode='r') as f:
        for file_list in f:
            col.append(file_list.split('\t'))
    for i in col:
        input.append(i[0])
        output.append(i[1].replace("\n", ""))
    bar = tqdm(total=len(input))
    for sentence in input:
        pred.append(gen_sentence(sentence, tok, model))
        bar.update(1)
    return input, output, pred


def convert_list_to_df(input, output, pred_list):
    row = []
    pred = ""
    for i in range(len(input)):
        input_str = input[i]
        output_str = output[i]
        batch_pred = pred_list[i]
        predict = [j for j in batch_pred if j != "<pad>" and j !=
                   "<sos>" and j != "<eos>" and j != "[CLS]" and j != "[unk]"]
        predict_str = "".join(predict).replace("#", "")
        row.append([input_str, output_str, predict_str])
    df = pd.DataFrame(row, columns=["input", "answer", "predict"])
    df = df.sort_values('input')
    return df


def main():

    print("preparing data...")
    path = "../data/data.tsv"
    src, trg, tmp = [], [], []
    with open(path, mode='r', encoding="utf-8") as f:
        for file in f:
            sentence = file.split("\t")
            tmp.append(sentence)

    random.shuffle(tmp)

    for sentence in tmp:
        src.append(sentence[0])
        trg.append(sentence[1].replace("\n", ""))

    src_tensors = tok.__call__(text=src, text_pair=trg, padding=True,
                               return_tensors='pt', return_attention_mask=False)
    trg_tensors = tok.__call__(text=trg, text_pair=src, padding=True,
                               return_tensors='pt', return_attention_mask=False)

    dataset = torch.utils.data.TensorDataset(src_tensors['input_ids'],
                                             trg_tensors['input_ids'])
    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size])

    batch_size = 32
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_data, batch_size, shuffle=True)

    print("building model...")
    OUTPUT_DIM = tok.vocab_size
    # OUTPUT_DIM = 3454
    ENC_EMB_DIM = 768
    DEC_EMB_DIM = 768
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    N_LAYERS = 4
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    enc = Encoder(ENC_EMB_DIM,
                  ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM,
                  DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    print(model)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 100
    clip = 1
    best_valid_loss = float('inf')
    best_model = None

    print("training...")
    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train(model, train_data_loader,
                           optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_data_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model

        print("-"*65)
        print(
            f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    torch.save(best_model.state_dict(),
               '../model/bert_embedded_seq2seq.pth')
    # model.apply(init_weights)

    # model.state_dict(torch.load("../model/bert_embedded_seq2seq.pth"))

    print("generating sentences...")
    path = "../data/test.tsv"
    test_input, test_output, test_pred = gen_sentence_list(
        model, path, tok)
    print(test_pred)

    path = "../data/train.tsv"
    train_input, train_output, train_pred = gen_sentence_list(
        model, path, tok)

    path = "../data/val.tsv"
    val_input, val_output, val_pred = gen_sentence_list(model, path, tok)

    train_df = convert_list_to_df(train_input, train_output, train_pred)
    val_df = convert_list_to_df(val_input, val_output, val_pred)
    test_df = convert_list_to_df(test_input, test_output, test_pred)

    df_s = pd.concat([train_df, test_df])
    df_s = df_s.sort_values("input")

    df_s.to_csv("../csv/result_bert_embedded_seq2seq.csv")
    df_result = df_s.groupby(["input", "predict"], as_index=False).agg({
        "answer": list
    })

    percentage, kinds, bleu = eval_score(df_result)
    print(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
    with open("score_bert_embedded_seq2seq.txt", mode="w") as f:
        f.write(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")

    print("done!!!")


if __name__ == "__main__":
    main()

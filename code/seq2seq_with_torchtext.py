from tqdm import tqdm
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


j_t = Tokenizer()

# 日本語を単語に分割したリストを返す関数


def tokenizer(text):
    return [tok for tok in j_t.tokenize(text, wakati=True)]


def choose_dataset(flag, SRC, TRG):
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
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 単語を分散表現に変換する
        embedded = self.embedding(src)
        # LSTMに単語を入力して、Encoderからの出力とする
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


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
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        # 出力をするためのやつ
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # 入力を整形する
        # input = input.unsqueeze(0)
        # 分散表現に変換
        embedded = self.embedding(input).unsqueeze(0)
        # print(embedded.size())
        # Decoderからの出力を得る
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # 正規化する
        prediction = self.fc_out(output.squeeze(0))

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

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(output, hidden, cell)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = (trg[t] if teacher_force else top1)

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train_model(model, iterator, optimizer, criterion, clip, TRG):
    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch.SRC
        trg = batch.TRG
        # src = torch.flip(src, [0, 1])
        src = src.to('cpu').detach().numpy().copy()
        src = np.flipud(src)
        src = torch.from_numpy(src.astype(np.int32)).clone()
        src = src.long().to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        # print(output.argmax(1))
        #print("output size:", output.size())
        #print("target size:", trg.size())
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # print("output size", output.size())
        # print("trg size: ", trg.size())
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_model(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.SRC
            trg = batch.TRG
            src = src.to('cpu').detach().numpy().copy()
            src = np.flipud(src)
            src = torch.from_numpy(src.astype(np.int32)).clone()
            src = src.long().to(device)

            output = model(src, trg)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    return elapsed_mins, elapsed_secs


def gen_sentence(sentence, src_field, trg_field, model, max_len=50):
    model.eval()

    tokens = [src_field.init_token] + \
        tokenizer(sentence) + [src_field.eos_token]

    src_index = [src_field.vocab.stoi[i] for i in tokens]
    src_index = np.flipud(np.array(src_index)).tolist()
    src_tensor = torch.LongTensor(src_index).unsqueeze(1).to(device)
    # print(src_tensor)
    # src_tensor = torch.flip(src_tensor, [0, 1])
    # print(src_tensor.size())
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_index = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_index[-1]]).to(device)
        # trg_tensor = torch.t(trg_tensor)
        with torch.no_grad():
            # print(trg_tensor.size())
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        # print(output)
        pred_token = output.argmax(1).item()
        # trg_index = np.array(trg_index).T.tolist()
        # print(trg_index)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

        trg_index.append(pred_token)

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_index]
    #if len(trg_tokens) == 2:
        # print(trg_tokens)
    #    trg_tokens = ["-"]
    #    trg_tokens = [src_field.init_token] + \
    #        trg_tokens + [src_field.eos_token]
    return trg_tokens


def gen_sentence_list(model, path, SRC, TRG):
    col, pred = [], []
    input, output = [], []
    with open(path, mode='r', encoding = "utf-8") as f:
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


def main():
    # pytorchのデータフィードの定義（重要！！）
    print("preparing data...")
    SRC = data.Field(sequential=True, tokenize=tokenizer,
                     init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = data.Field(sequential=True, tokenize=tokenizer,
                     init_token='<sos>', eos_token='<eos>', lower=True)

    train, val, test, filename = choose_dataset(False, SRC, TRG)

    # 辞書の作成
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # 各データをバッチ化する
    train_batch_size = 128
    test_batch_size = 32
    eval_batch_size = 32
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), sort=False,
                                                                 batch_sizes=(train_batch_size, eval_batch_size, test_batch_size), device=device)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 768
    DEC_EMB_DIM = 768
    ENC_HID_DIM = 1024
    DEC_HID_DIM = 1024
    N_LAYERS = 1
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    print(model)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    epochs = 100
    clip = 1
    best_model = None

    print("training...")
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train_model(model, train_iter, optimizer, criterion, clip, TRG)
        valid_loss = evaluate_model(model, val_iter, criterion)

        scheduler.step()
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            # torch.save(model.state_dict(), 'tut1-model.pt')

        print("-"*65)
        print(
            f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    torch.save(best_model.state_dict(), '../model/seq2seq.pth')

    # model.state_dict(torch.load("../model/seq2seq.pth"))
    print("generating sentence...")
    path = "../data/test.tsv"
    test_input, test_output, test_pred = gen_sentence_list(
        model, path, SRC, TRG)
    path = "../data/train.tsv"
    train_input, train_output, train_pred = gen_sentence_list(
        model, path, SRC, TRG)
    path = "../data/val.tsv"
    val_input, val_output, val_pred = gen_sentence_list(
        model, path, SRC, TRG)

    train_df = convert_list_to_df(train_input, train_output, train_pred)
    val_df = convert_list_to_df(val_input, val_output, val_pred)
    test_df = convert_list_to_df(test_input, test_output, test_pred)

    df_s = pd.concat([train_df, test_df]).sort_values('input').reset_index().drop(columns = ["index"])

    df_s.to_csv(filename)

    df_result = df_s.groupby(["input", "predict"], as_index=False).agg({
        "answer": list
    })

    percentage, kinds, bleu = eval_score(df_result)
    print(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
    with open("./score/score_seq2seq.txt", mode="w") as f:
        f.write(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
    print("done!")


if __name__ == "__main__":
    main()

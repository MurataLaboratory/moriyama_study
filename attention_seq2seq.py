from sklearn.model_selection import train_test_split
import os
import torch.optim as optim
import torch.nn as nn
import torch
import random
from sklearn.utils import shuffle

input_path = 'speak.txt'
output_path = 'res.txt'

input_data, output_data = [], []

with open(input_path, "r", encoding='utf-8') as f:
    input_data = f.readlines()

with open(output_path, "r", encoding='utf-8') as f:
    output_data = f.readlines()

input_len = len(input_data)
output_len = len(output_data)

char2id = {}

char2id.update({"<pad>": 0})

for input_chars, output_chars in zip(input_data, output_data):
    for c in input_chars:
        if not c in char2id:
            char2id[c] = len(char2id)
    for c in output_chars:
        if not c in char2id:
            char2id[c] = len(char2id)

print("vocab size: ", len(char2id))

input_id_data = []
output_id_data = []

for input_chars, output_chars in zip(input_data, output_data):
    input_id_data.append([char2id[c] for c in input_chars])
    output_id_data.append([char2id[c] for c in output_chars])


index_datasets_in_tmp = []
index_datasets_out_tmp = []

# 系列の長さの最大値を取得。この長さに他の系列の長さをあわせる
max_in_len = 0
max_out_len = 0
for input_d, out_d in zip(input_id_data, output_id_data):
    index_datasets_in_tmp.append(input_d)
    index_datasets_out_tmp.append(out_d)
    # 長さの最大値の計算
    if max_in_len < len(input_d):
        max_in_len = len(input_d)
    if max_out_len < len(out_d):
        max_out_len = len(out_d)

# 系列の長さを揃えるために短い系列にパディングを追加
index_datasets_input = []
index_datasets_out = []
for title in index_datasets_in_tmp:
    for i in range(max_in_len - len(title)):
        title.insert(0, 0)  # 前パディング
    index_datasets_input.append(title)

for title in index_datasets_out_tmp:
    for i in range(max_out_len - len(title)):
        title.insert(0, 0)  # 前パディング
    index_datasets_out.append(title)

train_x, test_x, train_y, test_y = train_test_split(
    index_datasets_input, index_datasets_out, train_size=0.7
)


# データのバッチ化

def train2batch(input_data, output_data, batch_size=100):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
        input_batch.append(input_shuffle[i:i+batch_size])
        output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch


# ハイパーパラメータの定義
embedding_dim = 200
hidden_dim = 120
BATCH_NUM = 100
EPOCH_NUM = 100
vocab_size = len(char2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder
# lstmではなく、GRUを使ってます


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)

        hs, h = self.gru(embedding)
        return hs, h


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)

        t_output = torch.transpose(output, 1, 2)

        s = torch.bmm(hs, t_output)
        attention_weight = self.softmax(s)

        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device)

        for i in range(attention_weight.size()[2]):

            unsq_weight = attention_weight[:, :, i].unsqueeze(2)

            weighted_hs = hs * unsq_weight

            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1)

            print(c.size())
            print(weight_sum.size())
            # cとweighted_sumの次元が違うのでエラーになる
            c = torch.cat([c, weight_sum], dim=1)
        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2)
        output = self.hidden2linear(output)

        return output, state, attention_weight


encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
attention_decoder = AttentionDecoder(
    vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)

criterion = nn.CrossEntropyLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
attention_decoder_optimizer = optim.Adam(
    attention_decoder.parameters(), lr=0.001)


all_losses = []

print("training...")

for epoch in range(1, EPOCH_NUM+1):
    epoch_loss = 0
    input_batch, output_batch = train2batch(
        train_x, train_y, batch_size=BATCH_NUM)
    for i in range(len(input_batch)):
        encoder_optimizer.zero_grad()
        attention_decoder_optimizer.zero_grad()

        input_tensor = torch.tensor(input_batch[i], device=device)
        output_tensor = torch.tensor(output_batch[i], device=device)

        hs, h = encoder(input_tensor)

        source = output_tensor[:, :-1]
        target = output_tensor[:, 1:]

        loss = 0
        decoder_output, _, attention_weight = attention_decoder(source, hs, h)

        for j in range(decoder_output.size()[1]):
            loss += criterion(decoder_output[:, j, :], target[:, j])

        epoch_loss += loss.item()

        loss.backward()

        encoder_optimizer.step()
        attention_decoder_optimizer.step()

    print("EPOCH %d: %.2f" % (epoch, epoch_loss))
    all_losses.append(epoch_loss)
    if epoch_loss < 0.1:
        break

prin("Done")

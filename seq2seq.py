import torch.optim as optim
import torch.nn as nn
import torch
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

input_path = 'speak.txt'
output_path = 'res.txt'

input_data = []
output_data = []

with open(input_path, "r", encoding='utf-8') as f:
    input_data = f.readlines()

with open(output_path, "r", encoding='utf-8') as f:
    output_data = f.readlines()

input_len = len(input_data)
output_len = len(output_data)


#　文字に対してIDを割り当てる
char2id = {}
# 長さを揃えるためのパディング文字を追加
char2id.update({"<pad>": 0})
for input_chars, output_chars in zip(input_data, output_data):
    for c in input_chars:
        if not c in char2id:
            char2id[c] = len(char2id)
    for c in output_chars:
        if not c in char2id:
            char2id[c] = len(char2id)

print("vocab size:", len(char2id))
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
    index_datasets_input, index_datasets_out, train_size=0.7)

# データのバッチ化


def train2batch(input_data, output_data, batch_size=100):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
        input_batch.append(input_shuffle[i:i+batch_size])
        output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch


"""
# ここからモデルを定義していきます
"""


# ハイパーパラメータの定義
embedding_dim = 200
hidden_dim = 120
BATCH_NUM = 100
EPOCH_NUM = 100
vocab_size = len(char2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        _, state = self.lstm(embedding)
        return state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequence, encoder_state):
        embedding = self.word_embeddings(sequence)
        output, state = self.lstm(embedding, encoder_state)
        output = self.hidden2linear(output)
        return output, state


# GPU使えるように。
encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

all_losses = []
print('training...')
for epoch in range(1, EPOCH_NUM+1):
    epoch_loss = 0
    input_batch, output_batch = train2batch(
        train_x, train_y, batch_size=BATCH_NUM)

    for i in range(len(input_batch)):
        # 勾配の初期化
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # データを行列に変換
        input_tensor = torch.tensor(input_batch[i], device=device)
        output_tensor = torch.tensor(output_batch[i], device=device)
        # encoderの順天板の学習
        encoder_state = encoder(input_tensor)
        # decoderの入力データ
        source = output_tensor[:, :-1]
        # decoderの教師データ
        target = output_tensor[:, 1:]
        loss = 0
        decoder_output, _ = decoder(source, encoder_state)

        for j in range(decoder_output.size()[1]):
            loss += criterion(decoder_output[:, j, :], target[:, j])

        epoch_loss += loss.item()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # 損失を表示
    print("Epoch %d: %.2f" % (epoch, epoch_loss))
    all_losses.append(epoch_loss)
    if epoch_loss < 1:
        break
print("Done")

path = __file__

torch.save(encoder.state_dict(), path)
torch.save(decoder.state_dict(), path)

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

path = "./data/data.tsv"
data = []
with open(path, mode="r", encoding="utf-8") as f:
    for file_list in f:
        data.append(file_list)

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

train, test = train_test_split(data, train_size = train_size, test_size = test_size)
test_size = int(len(test)/2)
val_size = len(test) - test_size
test, val = train_test_split(test, train_size = test_size, test_size = val_size)

print(len(train))
print(len(test))
print(len(val))

path = os.getcwd() + "/data/train.tsv"
print(path)

bar = tqdm(total = len(train))
with open(path, "w", encoding="utf-8") as f:
    for i in train:
        f.write(i)
        bar.update(1)

bar = tqdm(total = len(test))
path = os.getcwd() + "/data/test.tsv"
print(path)
with open(path, "w", encoding="utf-8") as f:
    for i in test:
        f.write(i)
        bar.update(1)

bar = tqdm(total = len(val))
path = os.getcwd() + "/data/val.tsv"
print(path)
with open(path, "w", encoding="utf-8") as f:
    for i in val:
        f.write(i)
        bar.update(1)
from sklearn.model_selection import train_test_split
import os

path = "./data/data.tsv"
data = []
with open(path, mode="r", encoding="utf-8") as f:
    for file_list in f:
        data.append(file_list)

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

train, test = train_test_split(data, train_size = train_size, test_size = test_size)

print(len(train))
print(len(test))

path = os.getcwd() + "/data/train.tsv"
print(path)
with open(path, "w", encoding="utf-8") as f:
    for i in train:
        print(i)
        f.write(i)

path = os.getcwd() + "/data/test.tsv"
print(path)
with open(path, "w", encoding="utf-8") as f:
    for i in test:
        print(i)
        f.write(i)
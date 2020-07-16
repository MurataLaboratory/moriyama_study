# 学習データで色々やってみるプログラム
import os

# 空白で区切って単語ごとに分ける
def split_word(word_list):
    return word_list.split(' ')

path = "narrative-response_train.txt"

list1, list2 = [], []
with open(path, encoding= "utf-8")as f:
    while True:
        l = f.readline()
        if l:
            # Tabについて分割
            s = l.split('\t')
            # 語りと応答について分割
            list1.append(s[0])
            # 謎に改行コードが入ったので消しておく
            list2.append(s[1].rstrip('\n'))
        else:
            break

print(split_word(list1[3]))
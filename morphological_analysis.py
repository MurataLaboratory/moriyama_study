# 学習データで色々やってみるプログラム
import os
import spacy
import torch
from torchtext import data, datasets
from torchtext.data import Field

# 日本語の学習済み統計モデルのロード
nlp = spacy.load('ja')
doc = 'spaCyはオープンソースの自然言語処理ライブラリです。学習済みの統計モデルと単語ベクトルが付属しています。'

# 何してるかよくわかってないけど単語区切りにしてくれるみたい


def tokenize_ja(text):
    return [tok.text for tok in nlp.tokenizer(text)]


# 学習で使うデータフィードの定義
TEXT = Field(tokenize=tokenize_ja,
             init_token='<sos>',
             eos_token='<eos>',
             lower=True)

# split_file_with_tab(path)
print(type(doc))
print(tokenize_ja(doc))
print(vars(TEXT))

path1 = 'speak.txt'

# 学習をするテキストに対してデータフィードの適用をする
lang = datasets.LanguageModelingDataset(path=path1,
                                        text_field=TEXT)

print(lang)

examples = lang.examples
print("Number of tokens: ", len(examples[0].text))
print("\n")
print("Print first 100 tokens: ", examples[0].text[:100])
print("\n")
print("Print last 10 tokens: ", examples[0].text[-10:])

# ヴォキャブラリーの作成
TEXT.build_vocab(lang)
vocab = TEXT.vocab
print("Vocabulary size: ", len(vocab))
print("10 most frequent words: ", vocab.freqs.most_common(10))
print("First 10 words: ", vocab.itos[0:10])
print("First 10 words of text data: ", lang.examples[0].text[:10])
print("Index of the first word: ", vocab.stoi[lang.examples[0].text[0]])

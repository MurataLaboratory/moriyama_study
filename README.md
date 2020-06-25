# 研究メモを書いていこうと思います

## 調べること

形態素解析解析とかの学習データの処理の確認。
データをいかにして読み込ませるか。

## ゴール

現状のゴールは Transformer とか Bert を目標にする

- 文献

  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### 6/18

ゼロから DL の RNN の章の読了

LSTM を始めた

### 6/25

６章 LSTM の章が終わった

LSTM におけるモデルの精度向上手段などについて学んだ。

- 有効な手法

  ドロップアウトの追加（時間方向には繋げない）

  LSTM 層の追加

  Affine レイヤと Embedding レイヤにおける重み共有

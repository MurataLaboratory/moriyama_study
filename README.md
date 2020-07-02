# 研究メモを書いていこうと思います

## 調べること

形態素解析解析とかの学習データの処理の確認。
データをいかにして読み込ませるか。

## ゴール

現状のゴールは Transformer とか Bert を目標にする

応答を生成したい
これまでの手法の改良を目指す(BERT,転移学習)
seq2seq ベースの生成方法の改良
事前学習　分散表現
正解の応答が決まっている
システムとしてよりユーザーに傾聴を届ける

- 文献

  [系列変換モデルに基づく傾聴的な応答表現の生成](https://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/P7-17.pdf)

  [語りの傾聴において表出する応答データの拡充](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P1-33.pdf)

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

７章を読み始めた

前の章で保存しておいたモデルの重みを使って文章の生成をためした。

### 7/2

7 章 seq2seq に入った

#### seq2seq とは

Encoder と Decoder の２つの RNN から構成される

ここでの出力は LSTM レイヤの最後の隠れ状態になる。これに必要な情報が入っている。（固定長のベクトルになる）

Encoder は入力情報をエンコードして Decoder はそれをもとに出力を生成する

#### seq2seq を改良したい

1.  入力データを反転させる

なぜうまく行くのかはわかっていないけど大体の場合うまく行くらしい

2. 覗き見

今のモデルではデコーダのの最初の LSTM&Affine しか受け取れないので全体に行き渡るようにする
(Peekydeocder,PeekySeq2seq に実装済み)

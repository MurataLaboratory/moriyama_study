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

<details>

<summary>文献</summary>

[系列変換モデルに基づく傾聴的な応答表現の生成](https://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/P7-17.pdf)

[語りの傾聴において表出する応答データの拡充](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P1-33.pdf)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

</details>

<details>

<summary>6/18</summary>

ゼロから DL の RNN の章の読了

LSTM を始めた

</details >

<details>
<summary> 6/25</summary>

６章 LSTM の章が終わった

LSTM におけるモデルの精度向上手段などについて学んだ。

- 有効な手法

  ドロップアウトの追加（時間方向には繋げない）

  LSTM 層の追加

  Affine レイヤと Embedding レイヤにおける重み共有

７章を読み始めた

前の章で保存しておいたモデルの重みを使って文章の生成をためした。

</details>

<details>

<summary>7/2 </summary>

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

</details>

<details>

<summary>7/9</summary>

ゼロから作る DL を読了した

イマイチ式とコードのイメージがつながらない

中間発表用のスライドを作ってました。なかなか難しい...

[中間発表用のスライドはこちら](https://kosenjp-my.sharepoint.com/:p:/g/personal/31540_toyota_kosen-ac_jp/ERB6GWA25h9EqEwibb21uE4B8uzeLus7C9gXFM_P-c1mYw?e=u1dPOk)

- seq2seq の問題点

固定長のベクトルを返す。（入力の長さに関わらず）

- Encoder を改良する

1. LSTM の重みを最後のものだけではなく*すべて*利用する

隠れ層の重みをすべて取り出して Encoder に渡してあげる

隠れ層の重みの中には入力された単語の情報がおおく含まれるため

- Decoder を改良する

1. 各単語ベクトルに対して重み付き和を計算する

2. 内積を用いてベクトル間の類似度を計算する

これらの層を組み合わせることで Attention を実現する

</details>

<details>
<summary>7/16</summary>

pytorchのサンプルを動かすときのspaCyのモデルは管理者権限でインストールする

発表用のプレゼンを完成させた

そろそろ pytorch を使ってモデルを作っていきたいので、まずは、前処理について調べている。

学習データは、タブ区切りで，左側が語り，右側が傾聴応答．語り，応答とも半角スペースで形態素に区切っています．

</details>

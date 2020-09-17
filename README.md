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

[github of Transformer](https://github.com/huggingface/transformers)

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

pytorch のサンプルを動かすときの spaCy のモデルは管理者権限でインストールする

発表用のプレゼンを完成させた

そろそろ pytorch を使ってモデルを作っていきたいので、まずは、前処理について調べている。

学習データは、タブ区切りで，左側が語り，右側が傾聴応答．語り，応答とも半角スペースで形態素に区切ってある．

別のファイル（morphological_analysis.py)をつくって前処理の練習をしてる

</details>

<details>

<summary>中間発表振り返り</summary>

質問

1. パフォーマンスはどうやって図るの？

   完全一致率->正解の応答と一致した生成された応答の数

   BLUE->生成された応答と正解がどの程度似ているかの指標

   種類数->生成された応答の種類

2. 系列変換ってどれのこと？

   seq2seq と Transformer の部分のとこで使われている（BERT はどうなんだろ）

</details>

<details>
<summary>7/30</summary>

- やったこと

テキストファイルを seq2seq で読み込ませるための前処理

見た感じうまく行ったので、seq2seq を動かせるようにしたい

ソースコードは morphological_analysis.py にある

呼び掛けと応答を別のファイルに保存してから、前処理をかけるように変更した

訓練用データとテスト用のデータへの分割に sklearn のものを使ってみる

- 参考リンク

[PyTorch で Seq2Seq を実装してみた](https://qiita.com/m__k/items/b18756628575b177b545)

前処理の参考

[Torchtext Tutorial 01: Getting Started](https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb)

BERT と Attention の解説記事

[自然言語処理の王様「BERT」の論文を徹底解説](https://qiita.com/omiita/items/72998858efc19a368e50#10-%E8%A6%81%E7%B4%84)

[深層学習界の大前提 Transformer の論文解説！](https://qiita.com/omiita/items/07e69aef6c156d23c538)

</details>

<details>
<summary>8/16</summary>

- 今日したこと

  サンプルの seq2seq はめんどくさそうなので自前でモデル部分だけ pytorch を使うように方向転換した

  seq2seq は一応動作するところまではできた。

- これから

  プロポーザルの修正と まとめ WORD を作成する。

  スライドに追加する画像（イラスト）と系列変換の部分の追加

</details>

<details>
<summary>9/10</summary>

- 今日したこと

  Attention を使った seq2seq を実装（丸写し）した

  Attention はまだデバックしていないです

- これからの予定

モデルを読み込んで、応答の生成をするスクリプトを作成する

BLUE などのスコアの計算もしていきたい

- 参考文献

[PyTorch で Attention Seq2Seq を実装してみた](https://qiita.com/m__k/items/646044788c5f94eadc8d)

</details>

<details>
<summary>9/17</summary>

- 今日したこと

コードを colabolatory に移植した。

seq2seq の場合の生成が一応できた

[結果はこちらです](https://drive.google.com/drive/folders/1wlw_0E57uI_qXNbg4HnL0xcrFFQjyb_S)

ファイル名は seq2seq.csv

- これからすること

スコアの計測

正しく生成できているかわからないので確認する

Attention 　 seq2seq も実行して生成してみる

</details>

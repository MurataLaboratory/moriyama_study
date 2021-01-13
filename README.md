# 研究メモを書いていこうと思います

## 現状の課題

Transformerにおける文章生成の関数とハイパーパラメータの調整

## 各ディレクトリについて

- code

各モデル（seq2seq,transformer)を学習させて文章を生成させるためのプログラムがあります

- csv

各モデルで文章生成をした結果と評価した結果が入ってます

- data.zip

モデルを学習させるためのデータが入ってます。`one_`で始まるデータは呼びかけと応答が1対1で対応してあります。

- evaluate.ipynb

評価するためのプログラム。文章生成したcsvファイルから、一致率と種類数とBLEUを測定して表示してくれる

- make_tsv.ipynb

データを改造するためのやつ

<details>

<summary>もらったデータをすべて使ったときのスコア</summary>

## seq2seqのスコア

  |完全一致率(%)|種類数|BLUE|
  |:--:|:--:|:--:|
  |12.48|40|0.05|


## Transformerのスコア

|完全一致率(%)|種類数|BLEU|
|:--:|:--:|:--:|
|27.84|506|0.08|

## Seq2seq + Bertの埋め込み層のスコア

|完全一致率(%)|種類数|BLEU|
|:--:|:--:|:--:|
|10.13|41|0.03|

## Transformer + Bertの埋め込み層のスコア

|完全一致率(%)|種類数|BLEU|
|:--:|:--:|:--:|
|23.61|589|0.07|

</details>

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

<details>
<summary>10/6</summary>

- 今日したこと

seq2seq の処理の確認

torchtext を使ってテキストデータの処理をした

参考: [torchtext で簡単に Deep な自然言語処理](https://qiita.com/itok_msi/items/1f3746f7e89a19dafac5)

- これからすること

いったん seq2seq の問題点を確認した後に、transformer のモデルを作成したい

ちゃんと生成されているか確認する

</details>

<details>

<summary>10/07</summary>

- 今日したこと

Attention seq2seq の実装

torchtext を使って、文章の前処理をした。

開始文字を S、終了文字を E にした。

参考にしている pytorch のチュートリアルではうまく行かなそうなので他のチュートリアルのやり方で試してみる

- これからすること

モデルの学習に必要な行列にテキストデータを変換して、学習させる。

</details>

<details>

<summary>10/08</summary>

- 今日したこと

  pytorch に実装されている Attention モデルを試した。

  torchtext を使った前処理をして、モデルの訓練をさせる予定。

  前処理はうまくいったけど訓練をするときにエラーをはいたので解決したい。

  途中で colaboratory が動かなくなったので seq2seq のスコアを計算してました

- これからすること

attention seq2seq の学習の実行（デバッグ）

</details>

<details>
<summary>10/13</summary>

- 今日したこと

  Attention Seq2seqの学習が動作した。

  pytorchでBERTを使うときの参考になりそうな記事を探していた。

  [日本語BERTモデルをPyTorch用に変換してfine-tuningする with torchtext & pytorch-lightning](https://radiology-nlp.hatenablog.com/entry/2020/01/18/013039)

  [Pretrained models -Hugging face](https://huggingface.co/transformers/pretrained_models.html)

  [cl-tohoku/bert-japanese](https://github.com/cl-tohoku/bert-japanese)

  [日本語BERTモデルに、センター試験や文章生成をやらせてみる](https://qiita.com/jun40vn/items/6458eb3a5301602d7092)

- これからすること

  Attention Seq2seqを使って文章を生成して、結果を確認する。

  pytorchを使ってBERTのfine Turningのやり方を知らべる

</details>

<details>
<summary>10/14</summary>

- 今日したこと

Transformerでの文章生成部分の作成。生成は正しくできてそう。

一通り生成できたと思います

- これからすること

生成結果を使って、スコアを計算するスクリプトを書く

</details>

<details>
<summary>10/13</summary>

- 今日したこと

Attention Seq2seqの学習と文章生成。結果は`result_transformer.csv`にあります。

各スコアを計算するためのスクリプトの作成

bleuのスコアがどうしても0に近い値になってしまう。結果は`score_seq2seq.csv`にあります。

- これからすること

BERTを使って文章を生成する。

BLEUのスコアについて考察する

</details>

<details>
<summary>10/20</summary>

- 今日したこと

BERTのモデルの作成

生成に関する論文

[BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://arxiv.org/abs/1902.04094)


- これからすること

参考にしているものがテキストの分類タスク用なので文章生成をするように変更する

引き続きBERTモデルの作成を続ける

</details>

<details>
<summary>10/21</summary>

- 今日したこと

BERTモデルの作成をした。

参考資料は分類用でテキストの変換などがどうすればいいのかわからないので別の方法を試す。

生成は`['MASK']`を連続で並べてやればできる。

- これからすること

BERTの重みを使ってseq2seqを初期化して学習させる。

</details>

<details>
<summary>10/22</summary>

- 今日したこと

ハイパーパラメータをseq2seqとtransformerでそろえた。

BERTの重みを取り出すことができた。

- これからすること

未知語の処理を考える。

BERTの重みでEmbeddingの初期化をする方法を考える。

</details>

<details>
<summary>10/27</summary>

- 今日したこと

生成された文が空のときに、2番目に推測される単語で補完するように改良した

中間発表のレジュメをかいてました。

- これからすること

生成された文の確認とBERTを使ってEmbeddingを初期化する方法を考える

</details>

<details>
<summary>10/28</summary>

- 今日したこと

Seq2seqとTransformerにおける学習の評価と中間発表用のWORDの作成

- これからすること

BERTから分散表現を取り出す。

中間発表のスライドの作成。

</details>

<details>
<summary>10/29</summary>

- 今日したこと

Transformerを使って学習を実行できた。

生成文を変換する関数を見直した。辞書には問題はなさそうなので、吐き出す確率の番号がおかしいと感じた。

入力文や応答に対する翻訳があってない（おかしい）。

seq2seqのプログラムは`seq2seq_with_torchtext.ipynb`、Attention付きseq2seqのプログラムは`attention=seq2seq.ipynb`にあります。

Multi30kを使った英独の翻訳データセットを使って3エポックだけモデルを訓練して文を生成した。

実行結果

```
入力
tensor([   2,    8,   36,   22,  245,   31,   12,   24,  122,   27,   14, 2047, 9,   35,    8,   16,   99,  290,    4,    3,    1,    1,    1,    1, 1,    1,    1,    1])
['<sos>', 'eine', 'gruppe', 'von', 'kindern', 'sitzt', 'auf', 'dem', 'boden', 'vor', 'einer', 'ziegelwand', ',', 'während', 'eine', 'frau', 'sie', 'beobachtet', '.', '<eos>']
答え
tensor([   2,    4,   38,   12,   63,  150,    8,    7,  259,  236,    4,  291, 108,   28,    4,   14, 1725,  155,    5,    3,    1,    1,    1,    1, 1,    1,    1])
['<sos>', 'a', 'group', 'of', 'children', 'sit', 'on', 'the', 'floor', 'against', 'a', 'brick', 'wall', 'while', 'a', 'woman', 'observes', 'them', '.', '<eos>']
予測
tensor([[ 0], [ 4], [ 9], [ 6], [ 4], [ 5], [ 3], [ 4], [39], [ 5], [ 3], [ 5], [ 5], [ 3], [ 4], [ 5], [ 3], [ 5], [ 5], [ 3], [ 5], [ 3], [ 5], [ 3], [ 5], [ 5], [ 3]])
['a', 'man', 'in', 'a', '.']

入力
tensor([  2,   5,  70,  32,  69,  20, 222, 140,   4,   3,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1])
['<sos>', 'ein', 'kleiner', 'hund', 'springt', 'im', 'sand', 'herum', '.', '<eos>']
答え
tensor([  2,   4,  70,  35,  92, 124,   7, 211,   3,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1])
['<sos>', 'a', 'small', 'dog', 'jumping', 'along', 'the', 'sand', '<eos>']
予測
tensor([[ 0], [ 4], [ 9], [ 6], [ 4], [ 9], [ 6], [23], [ 5], [ 3], [ 4], [ 9], [ 4], [ 9], [ 4], [ 9], [ 6], [ 4], [ 4], [ 4], [ 9], [ 6], [ 4], [ 9], [ 6], [ 5], [ 3]])
['a', 'man', 'in', 'a', 'man', 'in', 'shirt', '.']
```

- これからすること

中間発表のスライドの作成

</details>

<details>
<summary>11/4</summary>

- 今日したこと

スライドとワードの作成

作成している間にTransformerの学習を進めてました

BERTのEmbeddingを取り出すことができた

- これからすること

スライドの見直しとBERTの分散表現を用いた学習をしたい

</details>

<details>
<summary>11/10</summary>

- 今日したこと

Transformerモデルの学習部分と文章に変換する部分の確認（これと言ってミスは見当たらなかった)

pytorchのテンプレートでは、言語モデルを作っていた。これでも生成自体はできそうだと感じた。

ある程度の文章は生成できました

- これからすること

一旦、bertの分散表現を使ったseq2seqを作ります

</details>

<details>
<summary>11/17</summary>

- 今日したこと

bertを埋め込みを使ったseq2seqとTransformerの作成

プログラムの見直し

- これからすること

Transformer側のDecoder部分の追加実装

</details>

<details>
<summary>11/18</summary>

- 今日したこと

TransformerとBERTの分散表現を使ったTransformerの学習の実行

- これからすること

生成された文章の確認とスコアの計測

</details>

<details>
<summary>11/19</summary>

- 今日したこと

一応やりたいことはすべて終わってスコアの計測も終わりました。

- これからすること

応答と呼びかけを1対1で対応させたデータを使ってモデルを訓練し、応答を確認してみます

</details>

<details>
<summary>11/24</summary>

- 今日したこと

文章生成用の関数で`target`を入れるのはおかしい気がしたので、入れない関数に変更した。これで文章を生成して様子を見ます。

- これからすること

transformerでの文章生成関数を作成する

</details>

<details>
<summary>11/26</summary>

- 今日したこと

seq2seqでの文章生成時に出てくる何もない文の原因究明

ハイパーパラメータ（ドロップアウト率:0.5->0.3, 隠れ層の数:768 -> 256)を変更して要図を見たけど改善されなかった

- これからすること

引き続き色々いじってみます

</details>

<details>
<summary>12/01</summary>

- 今日したこと

Transformer側での文章生成関数の実装

lossの可視化をできるようにした

seq2seqはかぶりのないデータセットを使っても何も返さない時があってので根本的になにかが間違っているのかもしれない

- これからすること

文章生成関数のデバック

</details>

<details>
<summary>12/08</summary>

- 今日したこと

Seq2seqで空白語が生成される問題を解決した。隠れ層の数を増やしたら解決した。

- これからすること

Transformerの文章生成関数を作る。今の所、学習はできているが文章を生成しているわけではない。

</details>

<details>
<summary>12/09</summary>

- 今日したこと

linuxでモデルの訓練をするようにした。動作は確認した。

- これからすること

モデルの学習を進める

</details>

<details>
<summary>12/15</summary>

- 今日したこと

学習プログラムのデバック

- これからすること

引き続き学習プログラムのデバックをしてスコアを計測していきたい

</details>

<details>
<summary>12/16</summary>

- 今日したこと

プログラムのデバック

文章生成の進捗を見れるようにプログレスバーを追加してみた

- これからすること

seq2seqで空白文字が生成されないようなパラメータ探し

</details>

<details>
<summary>12/17</summary>

- 今日したこと

seq2seqの学習。未だに何も言わない時がある

- これからすること

何も言われないことのないパラメータ探し

</details>

<details>
<summary>12/22</summary>

- 今日したこと

TransformerとBERTの埋め込み層を使ったSeq2seqの出力の確認。

Transformerは何も言わなくて、seq2seqは支離滅裂な文章が多かった。

論文を読んだら、入力文の反転が有効と書いてあったのでやってみようと思う

- これからすること

学習させます

</details>

<details>
<summary>12/24</summary>

- 今日したこと

Transformerのバグ取り

- これからすること

学習させる

</details>

<details>
<summary>01/07</summary>

- 今日したこと

モデルの学習

- これからすること

seq2seqで空白文字が出るのでプログラムを見直す

</details>

<details>
<summary>01/12</summary>

- 今日したこと

出力分がおかしかったのでdateloaderを見直したら、バグを見つけた。修正して学習し直す

- これからすること

概要を書く

</details>
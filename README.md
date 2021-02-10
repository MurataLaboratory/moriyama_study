# READ ME

## ディレクトリの内容

- code
 モデルの学習プログラムが入っているディレクトリ

## 実行手順

### パッケージのインスール

これだけ入れとけば全て動くはず

- pytorch
- torchtext
- janome
- transformers
- pandas
- numpy

### zipファイルの解凍

`data.zip`を解凍してください．

### make_data.py

`make_data.py`を実行してください

学習データ，テストデータ，検証データが生成されます

### 学習

codeディレクトリにseq2seq，tranformerが実装してあるので，実行してください．

テストデータに対する生成結果は，`/csv/test/`の中，訓練データに対する生成結果は`/csv/train/`の中にあります

スコアは`/code/score/`の中に保存されるはずです

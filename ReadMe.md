<!-- TOC -->

- [ライブラリの種類と用途](#ライブラリの種類と用途)
  - [データ加工](#データ加工)
  - [画像分析](#画像分析)
  - [機械学習](#機械学習)
  - [可視化](#可視化)
  - [ツール](#ツール)
- [インストール](#インストール)
  - [NumPy](#numpy)
  - [SciPy](#scipy)
  - [SciPy](#scipy-1)
  - [Pillow（PIL）](#pillowpil)
  - [scikit-learn](#scikit-learn)
  - [Tensor Flow](#tensor-flow)
    - [CUDA のインストール](#cuda-のインストール)
    - [cuDNN のインストール](#cudnn-のインストール)
    - [再度動作確認](#再度動作確認)
  - [Keras](#keras)
  - [Matplotlib](#matplotlib)
- [pandas](#pandas)
  - [おまじない](#おまじない)
    - [パッケージの読み込み](#パッケージの読み込み)
    - [共通設定](#共通設定)
      - [設定値の書き込み](#設定値の書き込み)
      - [設定値の確認](#設定値の確認)
      - [特定の処理だけ設定変更した状態で実行](#特定の処理だけ設定変更した状態で実行)
  - [データ型](#データ型)
    - [Python リストから生成](#python-リストから生成)
    - [NumPy 配列から生成](#numpy-配列から生成)
  - [I/O](#io)
    - [CSV](#csv)
      - [読み込み](#読み込み)
        - [ヘッダー](#ヘッダー)
        - [カラム名](#カラム名)
        - [インデックス名](#インデックス名)
        - [読み込む列を指定](#読み込む列を指定)
        - [読み込む行を指定](#読み込む行を指定)
        - [読み込む行数を指定](#読み込む行数を指定)
        - [読み込まない行を指定](#読み込まない行を指定)
        - [末尾から指定](#末尾から指定)
        - [読み込むデータ型を指定](#読み込むデータ型を指定)
        - [欠損値処理](#欠損値処理)
        - [読み込む文字コードを指定](#読み込む文字コードを指定)
        - [圧縮された CSV ファイルを読み込む](#圧縮された-csv-ファイルを読み込む)
        - [Web 上の CSV ファイルを読み込む](#web-上の-csv-ファイルを読み込む)
        - [時系列データ](#時系列データ)
      - [書き出し](#書き出し)
    - [TSV](#tsv)
      - [読み込み](#読み込み-1)
      - [書き出し](#書き出し-1)
    - [JSON](#json)
      - [読み込み](#読み込み-2)
      - [書き出し](#書き出し-2)
      - [フォーマット](#フォーマット)
    - [JSON 文字列の読み込み](#json-文字列の読み込み)
    - [JSON Lines](#json-lines)
    - [Excel 形式（xlsx）](#excel-形式xlsx)
      - [依存パッケージをインストール](#依存パッケージをインストール)
      - [読み込み](#読み込み-3)
      - [書き出し](#書き出し-3)
    - [pickle](#pickle)
      - [読み込み](#読み込み-4)
      - [書き出し](#書き出し-4)
    - [HTML](#html)
      - [Web ページ内の table（表）を読み込む](#web-ページ内の-table表を読み込む)
        - [依存パッケージをインストール](#依存パッケージをインストール-1)
        - [読み込み](#読み込み-5)
  - [データ抽出](#データ抽出)
    - [Dataframe の概要を確認](#dataframe-の概要を確認)
      - [情報表示・統計量](#情報表示・統計量)
      - [行名・列名、行数・列数・要素数](#行名・列名行数・列数・要素数)
        - [行名を変更](#行名を変更)
          - [列をインデックスとして使用](#列をインデックスとして使用)
          - [連番にリセットする](#連番にリセットする)
      - [行名・列名、行数・列数・要素数](#行名・列名行数・列数・要素数-1)
      - [欠損値の数](#欠損値の数)
      - [行を絞る](#行を絞る)
        - [先頭](#先頭)
        - [末尾](#末尾)
      - [インデックスから行と列を絞る（DataFrame）](#インデックスから行と列を絞るdataframe)
        - [行または列](#行または列)
        - [行と列](#行と列)
          - [添え字で辿っていく方法](#添え字で辿っていく方法)
      - [インデックスから行を絞る（Series）](#インデックスから行を絞るseries)
      - [条件に適合する行を抽出](#条件に適合する行を抽出)
        - [ブールインデックス](#ブールインデックス)
        - [query メソッド](#query-メソッド)
      - [条件に適合する列を抽出](#条件に適合する列を抽出)
        - [ブールインデックス](#ブールインデックス-1)
      - [欠損値を除去](#欠損値を除去)
      - [ソート](#ソート)
        - [行や列の値でソート](#行や列の値でソート)
        - [行や列のインデックスでソート](#行や列のインデックスでソート)
        - [ランダムソート](#ランダムソート)
  - [データ加工](#データ加工-1)
    - [追加](#追加)
      - [行の追加](#行の追加)
        - [append](#append)
        - [concat](#concat)
      - [列の追加](#列の追加)
        - [列名](#列名)
        - [assign](#assign)
        - [insert（位置を指定して追加）](#insert位置を指定して追加)
        - [concat（DataFrame に Series を列として追加）](#concatdataframe-に-series-を列として追加)
    - [要素に対する演算](#要素に対する演算)
    - [転置](#転置)
    - [値の置換](#値の置換)
      - [欠損値](#欠損値)
        - [欠損値に置き換える](#欠損値に置き換える)
        - [欠損値を置き換える](#欠損値を置き換える)
          - [定数で置換](#定数で置換)
          - [統計量で置換](#統計量で置換)
          - [前後の要素で置換](#前後の要素で置換)
    - [特徴量エンジニアリング](#特徴量エンジニアリング)
      - [ビニング処理（ビン分割）](#ビニング処理ビン分割)
      - [正規化](#正規化)
        - [最小値 0、最大値 1](#最小値-0最大値-1)
        - [平均 0、分散 1](#平均-0分散-1)
      - [カテゴリ変数のエンコーディング](#カテゴリ変数のエンコーディング)
  - [データ結合](#データ結合)
    - [inner join](#inner-join)
      - [複数キー](#複数キー)
    - [left join](#left-join)
    - [right join](#right-join)
    - [outer join](#outer-join)
  - [グループ化](#グループ化)
  - [時系列データ](#時系列データ-1)
    - [頻度の指定（オフセットエイリアス）](#頻度の指定オフセットエイリアス)
    - [ラグ](#ラグ)
    - [時系列データのグループ化](#時系列データのグループ化)
    - [時系列データの抽出](#時系列データの抽出)
  - [データ可視化](#データ可視化)
- [Numpy](#numpy)
- [Scipy](#scipy)
- [Pillow（PIL）](#pillowpil-1)
- [scikit-learn](#scikit-learn-1)
- [Keras](#keras-1)
- [TensorFlow](#tensorflow)
- [Matplotlib](#matplotlib-1)

<!-- /TOC -->

<a id="markdown-ライブラリの種類と用途" name="ライブラリの種類と用途"></a>

# ライブラリの種類と用途

<a id="markdown-データ加工" name="データ加工"></a>

## データ加工

- 組み込み関数

- NumPy
- SciPy
- Pandas

<a id="markdown-画像分析" name="画像分析"></a>

## 画像分析

- Pillow（PIL）

<a id="markdown-機械学習" name="機械学習"></a>

## 機械学習

- scikit-learn
- Tensor Flow
- Keras
- PySpark

<a id="markdown-可視化" name="可視化"></a>

## 可視化

- Matplotlib

<a id="markdown-ツール" name="ツール"></a>

## ツール

- Jupyter

<a id="markdown-インストール" name="インストール"></a>

# インストール

<a id="markdown-numpy" name="numpy"></a>

## NumPy

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install numpy

(myenv)$ python -c "import numpy as np" # エラーが出なければ正常にインストールされている
```

※以下のような警告が出ることがあるが、モジュールとして実行（ `python -m numpy.f2py` ）する場合は無視してよい

```
WARNING: The script f2py.exe is installed in 'C:\Users\y\AppData\Roaming\Python\Python38\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this
warning, use --no-warn-script-location.
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install numpy

(myenv)$ python -c "import numpy as np" # エラーが出なければ正常にインストールされている
```

<a id="markdown-scipy" name="scipy"></a>

## SciPy

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install scipy # NumPyもインストールされる

(myenv)$ python -c "from scipy import stats" # エラーが出なければ正常にインストールされている
```

※確認の際に、 `import scipy` とするとエラーとなるので注意（きちんとサブパッケージを import する必要がある）

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install scipy # NumPyもインストールされる

(myenv)$ python -c "from scipy import stats" # エラーが出なければ正常にインストールされている
```

<a id="markdown-scipy-1" name="scipy-1"></a>

## SciPy

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install pandas # NumPy, SciPyもインストールされる

(myenv)$ python -c "import pandas as pd" # エラーが出なければ正常にインストールされている
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install pandas # NumPy, SciPyもインストールされる

(myenv)$ python -c "import pandas as pd" # エラーが出なければ正常にインストールされている
```

<a id="markdown-pillowpil" name="pillowpil"></a>

## Pillow（PIL）

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install pillow # NumPyもインストールされる

(myenv)$ python -c "import PIL" # エラーが出なければ正常にインストールされている
```

※パッケージ名は Pillow に改名されているが、import 時には PIL を指定する必要がある

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install pillow # NumPyもインストールされる

(myenv)$ python -c "import PIL" # エラーが出なければ正常にインストールされている
```

<a id="markdown-scikit-learn" name="scikit-learn"></a>

## scikit-learn

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install scikit-learn # NumPy, SciPyもインストールされる

(myenv)$ python -c "import sklearn; sklearn.show_versions()" # 動作確認
```

```
System:
    python: 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]
executable: \path\to\python.exe
   machine: Windows-10

Python dependencies:
       pip: 20.1
setuptools: 41.2.0
   sklearn: 0.22.2.post1
     numpy: 1.18.3
     scipy: 1.4.1
    Cython: None
    pandas: None
matplotlib: None
    joblib: 0.14.1

Built with OpenMP: True
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install scikit-learn # NumPy, SciPyもインストールされる

(myenv)$ python -c "import sklearn; sklearn.show_versions()" # 動作確認
```

<a id="markdown-tensor-flow" name="tensor-flow"></a>

## Tensor Flow

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install tensorflow
(myenv)$ python -m pip install tensorflow-gpu

(myenv)$ python -c "import tensorflow as tf; print(tf.__version__)" # 動作確認
```

Tensor Flow のインストールは完了しているものの、GPU を利用する準備ができていない状態

```
Using TensorFlow backend.
2020-05-09 08:44:03.639530: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-05-09 08:44:03.702221: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2.2.0
```

<a id="markdown-cuda-のインストール" name="cuda-のインストール"></a>

### CUDA のインストール

- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) から、CUDA Toolkit 10.1 のインストーラーを取得する
  - [Installer for Windows 10 x86_64](http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe)

<a id="markdown-cudnn-のインストール" name="cudnn-のインストール"></a>

### cuDNN のインストール

- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
  - [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)
    - [cuDNN Library for Windows 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip)

<a id="markdown-再度動作確認" name="再度動作確認"></a>

### 再度動作確認

```
2020-05-09 09:06:08.391168: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2.2.0
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install tensorflow # tensorflow-gpuは非対応

(myenv)$ python -c "import tensorflow as tf; print(tf.__version__)" # 動作確認
```

<a id="markdown-keras" name="keras"></a>

## Keras

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install tensorflow tensorflow-gpu # バックエンドにTensor Flowを利用
(myenv)$ python -m pip install keras

(myenv)$ python -c "import keras; print(keras.__version__)" # 動作確認
```

```
Using TensorFlow backend.
2020-05-09 09:06:08.391168: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2.3.1
```

以下のような警告メッセージが出力されたら、[Tensor Flow](#Tensor Flow)の項を参考に、CUDA と cuDNN をセットアップする

```
2020-05-09 08:44:03.639530: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-05-09 08:44:03.702221: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install keras tensorflow # tensorflow-gpuは非対応

(myenv)$ python -c "import keras; print(keras.__version__)" # 動作確認
```

<a id="markdown-matplotlib" name="matplotlib"></a>

## Matplotlib

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install matplotlib # Numpyも依存関係で入る

(myenv)$ python -c "import matplotlib; print(matplotlib.__version__)" # 動作確認
```

```
3.2.1
```

---

（Mac）

```sh
$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install matplotlib

(myenv)$ python -c "import matplotlib; print(matplotlib.__version__)" # 動作確認
```

<a id="markdown-pandas" name="pandas"></a>

# pandas

<a id="markdown-おまじない" name="おまじない"></a>

## おまじない

<a id="markdown-パッケージの読み込み" name="パッケージの読み込み"></a>

### パッケージの読み込み

```py
import pandas as pd
import numpy as np
```

<a id="markdown-共通設定" name="共通設定"></a>

### 共通設定

<a id="markdown-設定値の書き込み" name="設定値の書き込み"></a>

#### 設定値の書き込み

```py
import pandas as pd

pd.options.display.max_columns = None # 全ての列を表示
pd.set_option('display.max_rows', None) # 全ての行を表示
pd.set_option('max_.*', 100) # 正規表現でも指定できるが、複数該当するとOptionErrorとなる

# 設定できる項目を確認する
import pprint
pprint.pprint(dir(pd.options))

# 既定値に戻す
pd.reset_option('display.max_columns')
pd.reset_option('max_col') # 複数該当した場合はその全てで既定値に戻す
pd.reset_option('^display', silent=True) # display.～をの全てで既定値に戻す
pd.reset_option('all', silent=True)
```

<a id="markdown-設定値の確認" name="設定値の確認"></a>

#### 設定値の確認

```py
import pandas as pd

print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_.*')) # 正規表現でも指定できるが、複数該当するとOptionErrorとなる

pd.describe_option() # 全件
pd.describe_option('max.*col') # 正規表現でフィルタリング
```

<a id="markdown-特定の処理だけ設定変更した状態で実行" name="特定の処理だけ設定変更した状態で実行"></a>

#### 特定の処理だけ設定変更した状態で実行

```py
import pandas as pd

pd.reset_option('display.max_rows')
print(pd.get_option('display.max_rows')) # 60

pd.options.display.max_rows = 10
print(pd.get_option('display.max_rows')) # 10

with pd.option_context('display.max_rows', 3): # withブロックの中だけ有効
    print(pd.get_option('display.max_rows')) # 3

print(pd.get_option('display.max_rows')) # 10
```

```
60
10
3
10
```

<a id="markdown-データ型" name="データ型"></a>

## データ型

- オブジェクト
  - 1 次元: シリーズ `pandas.Series`
  - 2 次元: データフレーム `panas.DataFrame`

DataFrame の列や Series は、以下のデータ型の値を取る

| dtype         | 対応する Python のデータ型 |
| ------------- | -------------------------- |
| object        | str                        |
| int64         | int                        |
| float64       | float                      |
| bool          | bool                       |
| datetime64    |                            |
| timedelta[ns] |                            |
| category      |                            |

<a id="markdown-python-リストから生成" name="python-リストから生成"></a>

### Python リストから生成

```py
import pandas as pd

lst = range(3) # Pythonのリスト

pd.Series(lst) # 1次元: シリーズ
pd.DataFrame(lst) # 2次元: データフレーム

df = pd.DataFrame(lst)
df.dtypes # データ型の確認
df = df.astype('float32') # データ型の変換
df.dtypes # 変換後のデータ型の確認
```

```
# 1次元: シリーズ
0    1
1    2
2    3
dtype: int64

# 2次元: データフレーム
   0
0  1
1  2
2  3

# データ型の確認
0    int64
dtype: object

# 変換後のデータ型の確認
0    float32
dtype: object
```

<a id="markdown-numpy-配列から生成" name="numpy-配列から生成"></a>

### NumPy 配列から生成

```py
import numpy as np
import pandas as pd

lst = np.arange(3) # NumPyの配列
idx = pd.Index(['r1', 'r2', 'r3'], name = 'index')
pd.Series(lst, index=idx) # 1次元: シリーズ

arr = np.arange(4).reshape(2, 2)
idx = pd.Index(['r1', 'r2'], name = 'index')
col = pd.Index(['c1', 'c2'], name= 'column')
pd.DataFrame(arr, index=idx, columns=col) # 2次元: データフレーム

lst = np.arange(4)
idx = pd.MultiIndex.from_product([['r1','r2'],['c1','c2']], names=('R','C'))
pd.Series(lst, index=idx) # 2次元: シリーズ

lst = np.arange(8)
idx = pd.MultiIndex.from_product([['x1','x2'],['y1','y2'],['z1','z2']], names=('X','Y','Z'))
pd.Series(lst, index=idx) # 3次元: シリーズ

arr = np.arange(16) .reshape(4,4)
idx = pd.MultiIndex.from_product( [['x1','x2'],['y1','y2']], names=('X','Y'))
col = pd.MultiIndex.from_product( [['z1','z2'],['w1','w2']], names=('Z','W'))
pd.DataFrame(arr, index=idx, columns=col) # 4次元: データフレーム
```

```
# 1次元: シリーズ
index
r1    0
r2    1
r3    2
dtype: int32

# 2次元: データフレーム
column  c1  c2
index
r1       0   1
r2       2   3

# 2次元: シリーズ
R   C
r1  c1    0
    c2    1
r2  c1    2
    c2    3
dtype: int32

# 3次元: シリーズ
X   Y   Z
x1  y1  z1    0
        z2    1
    y2  z1    2
        z2    3
x2  y1  z1    4
        z2    5
    y2  z1    6
        z2    7
dtype: int32

# 4次元: データフレーム
Z      z1      z2
W      w1  w2  w1  w2
X  Y
x1 y1   0   1   2   3
   y2   4   5   6   7
x2 y1   8   9  10  11
   y2  12  13  14  15
```

<a id="markdown-io" name="io"></a>

## I/O

<a id="markdown-csv" name="csv"></a>

### CSV

<a id="markdown-読み込み" name="読み込み"></a>

#### 読み込み

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv.csv')
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv')
print(df2)
print(df2.columns)
```

```
   11  12  13  14
0  21  22  23  24
1  31  32  33  34

Index(['11', '12', '13', '14'], dtype='object')

   c1  c2  c3  c4
0  11  12  13  14
1  21  22  23  24
2  31  32  33  34

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')
```

<a id="markdown-ヘッダー" name="ヘッダー"></a>

##### ヘッダー

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv.csv', header=None)
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', header=None)
print(df2)
print(df2.columns)

df1 = pd.read_csv('data/pandas/read_csv.csv', header=0)
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', header=0)
print(df2)
print(df2.columns)

df1 = pd.read_csv('data/pandas/read_csv.csv', header=1) # 途中の行から読み込む
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', header=1)
print(df2)
print(df2.columns)
```

```
    0   1   2   3
0  11  12  13  14
1  21  22  23  24
2  31  32  33  34

Int64Index([0, 1, 2, 3], dtype='int64')

    0   1   2   3
0  c1  c2  c3  c4
1  11  12  13  14
2  21  22  23  24
3  31  32  33  34

Int64Index([0, 1, 2, 3], dtype='int64')

   11  12  13  14
0  21  22  23  24
1  31  32  33  34

Index(['11', '12', '13', '14'], dtype='object')

   c1  c2  c3  c4
0  11  12  13  14
1  21  22  23  24
2  31  32  33  34

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')

   21  22  23  24
0  31  32  33  34

Index(['21', '22', '23', '24'], dtype='object')

   11  12  13  14
0  21  22  23  24
1  31  32  33  34

Index(['11', '12', '13', '14'], dtype='object')
```

<a id="markdown-カラム名" name="カラム名"></a>

##### カラム名

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv.csv', names=('C1', 'C2', 'C3', 'C4'))
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', names=('C1', 'C2', 'C3', 'C4'))
print(df2)
print(df2.columns)
```

```
   C1  C2  C3  C4
0  11  12  13  14
1  21  22  23  24
2  31  32  33  34

Index(['C1', 'C2', 'C3', 'C4'], dtype='object')

   C1  C2  C3  C4
0  c1  c2  c3  c4
1  11  12  13  14
2  21  22  23  24
3  31  32  33  34

Index(['C1', 'C2', 'C3', 'C4'], dtype='object')
```

<a id="markdown-インデックス名" name="インデックス名"></a>

##### インデックス名

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_index.csv', header=None, index_col=0)
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_index.csv', header=None, index_col=1)
print(df2)
print(df2.columns)
```

```
     1   2   3   4
0
r1  11  12  13  14
r2  21  22  23  24
r3  31  32  33  34

Int64Index([1, 2, 3, 4], dtype='int64')

     0   2   3   4
1
11  r1  12  13  14
21  r2  22  23  24
31  r3  32  33  34

Int64Index([0, 2, 3, 4], dtype='int64')
```

<a id="markdown-読み込む列を指定" name="読み込む列を指定"></a>

##### 読み込む列を指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv.csv', header=None, usecols=[2,3])
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', usecols=['c1', 'c2'])
print(df2)
print(df2.columns)
df3 = pd.read_csv('data/pandas/read_csv_header.csv', usecols=lambda x: x not in ['c2'])
print(df3)
print(df3.columns)
df4 = pd.read_csv('data/pandas/read_csv_header.csv', index_col='c1', usecols=lambda x: x not in ['c2'])
print(df4)
print(df4.columns)
```

```
    2   3
0  13  14
1  23  24
2  33  34

Int64Index([2, 3], dtype='int64')

   c1  c2
0  11  12
1  21  22
2  31  32

Index(['c1', 'c2'], dtype='object')

   c1  c3  c4
0  11  13  14
1  21  23  24
2  31  33  34

Index(['c1', 'c3', 'c4'], dtype='object')

    c3  c4
c1
11  13  14
21  23  24
31  33  34

Index(['c3', 'c4'], dtype='object')
```

<a id="markdown-読み込む行を指定" name="読み込む行を指定"></a>

##### 読み込む行を指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_header.csv', skiprows=lambda x: x not in [0, 2]) # ヘッダーと2行目以外をスキップ
print(df1)
print(df1.columns)
```

```
   c1  c2  c3  c4
0  21  22  23  24

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')
```

<a id="markdown-読み込む行数を指定" name="読み込む行数を指定"></a>

##### 読み込む行数を指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_header.csv',  nrows=2) # 最初の2行を読み込む
print(df1)
print(df1.columns)
```

```
   c1  c2  c3  c4
0  11  12  13  14
1  21  22  23  24

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')
```

<a id="markdown-読み込まない行を指定" name="読み込まない行を指定"></a>

##### 読み込まない行を指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_header.csv', skiprows=2) # 2行分スキップ
print(df1)
print(df1.columns)
df2 = pd.read_csv('data/pandas/read_csv_header.csv', skiprows=[1]) # 1行目をスキップ
print(df2)
print(df2.columns)
df3 = pd.read_csv('data/pandas/read_csv_header.csv', skiprows=[1,2]) # 1,2行目をスキップ
print(df3)
print(df3.columns)
```

```
   21  22  23  24
0  31  32  33  34

Index(['21', '22', '23', '24'], dtype='object')

   c1  c2  c3  c4
0  21  22  23  24
1  31  32  33  34

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')

   c1  c2  c3  c4
0  31  32  33  34

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')
```

<a id="markdown-末尾から指定" name="末尾から指定"></a>

##### 末尾から指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_header.csv', skipfooter=1, engine='python')
print(df1)
print(df1.columns)
```

```
   c1  c2  c3  c4
0  11  12  13  14
1  21  22  23  24

Index(['c1', 'c2', 'c3', 'c4'], dtype='object')
```

<a id="markdown-読み込むデータ型を指定" name="読み込むデータ型を指定"></a>

##### 読み込むデータ型を指定

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_dtype.csv')
print(df1)
print(df1.dtypes)
print(df1.applymap(type))

df1_astype = df1.astype({'col3': str})
print(df1_astype)
print(df1_astype.dtypes)
print(df1_astype.applymap(type))

df2 = pd.read_csv('data/pandas/read_csv_dtype.csv', dtype=str)
print(df2)
print(df2.dtypes)
print(df2.applymap(type))

df3 = pd.read_csv('data/pandas/read_csv_dtype.csv', dtype={'col3': str, 'col4': str})
print(df3)
print(df3.dtypes)
print(df3.applymap(type))

df4 = pd.read_csv('data/pandas/read_csv_dtype.csv', dtype={2: str, 3: str})
print(df4)
print(df4.dtypes)
print(df4.applymap(type))
```

- 元の CSV ファイル `read_csv_dtype.csv`

```csv
col1,col2,col3,col4
a11,12,013,14
b21,22,023,24
c31,32,033,34
```

- df1

```
  col1  col2  col3  col4
0  a11    12    13    14
1  b21    22    23    24
2  c31    32    33    34

col1    object
col2     int64
col3     int64
col4     int64
dtype: object

            col1           col2           col3           col4
0  <class 'str'>  <class 'int'>  <class 'int'>  <class 'int'>
1  <class 'str'>  <class 'int'>  <class 'int'>  <class 'int'>
2  <class 'str'>  <class 'int'>  <class 'int'>  <class 'int'>
```

- df1_astype

```
  col1  col2 col3  col4
0  a11    12   13    14
1  b21    22   23    24
2  c31    32   33    34

col1    object
col2     int64
col3    object
col4     int64
dtype: object

            col1           col2           col3           col4
0  <class 'str'>  <class 'int'>  <class 'str'>  <class 'int'>
1  <class 'str'>  <class 'int'>  <class 'str'>  <class 'int'>
2  <class 'str'>  <class 'int'>  <class 'str'>  <class 'int'>
```

- df2

```
  col1 col2 col3 col4
0  a11   12  013   14
1  b21   22  023   24
2  c31   32  033   34

col1    object
col2    object
col3    object
col4    object
dtype: object

            col1           col2           col3           col4
0  <class 'str'>  <class 'str'>  <class 'str'>  <class 'str'>
1  <class 'str'>  <class 'str'>  <class 'str'>  <class 'str'>
2  <class 'str'>  <class 'str'>  <class 'str'>  <class 'str'>
```

- df3

```
  col1  col2 col3 col4
0  a11    12  013   14
1  b21    22  023   24
2  c31    32  033   34

col1    object
col2     int64
col3    object
col4    object
dtype: object

            col1           col2           col3           col4
0  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
1  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
2  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
```

- df4

```
  col1  col2 col3 col4
0  a11    12  013   14
1  b21    22  023   24
2  c31    32  033   34

col1    object
col2     int64
col3    object
col4    object
dtype: object

            col1           col2           col3           col4
0  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
1  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
2  <class 'str'>  <class 'int'>  <class 'str'>  <class 'str'>
```

<a id="markdown-欠損値処理" name="欠損値処理"></a>

##### 欠損値処理

既定では `'', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'` を欠損値として扱う

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv_na.csv', na_values='-')
print(df1)
print(df1.isnull())

# 既定の欠損値ではなく指定した値のみを欠損値として扱う
df2 = pd.read_csv('data/pandas/read_csv_na.csv', na_values=['-', 'NA'], keep_default_na=False)
print(df2)
print(df2.isnull())

# 欠損値処理をしない
df3 = pd.read_csv('data/pandas/read_csv_na.csv', na_filter=False)
print(df3)
print(df3.isnull())
```

- df1

```
  col1  col2  col3  col4
0  a11    12   NaN   NaN
1  b21    22   NaN  24.0
2  c31    32  33.0  34.0

    col1   col2   col3   col4
0  False  False   True   True
1  False  False   True  False
2  False  False  False  False
```

- df2

```
  col1  col2 col3  col4
0  a11    12  NaN   NaN
1  b21    22       24.0
2  c31    32  033  34.0

    col1   col2   col3   col4
0  False  False   True   True
1  False  False  False  False
2  False  False  False  False
```

- df3

```
  col1  col2 col3 col4
0  a11    12    -   NA
1  b21    22        24
2  c31    32  033   34

    col1   col2   col3   col4
0  False  False  False  False
1  False  False  False  False
2  False  False  False  False
```

<a id="markdown-読み込む文字コードを指定" name="読み込む文字コードを指定"></a>

##### 読み込む文字コードを指定

UTF-8 以外のエンコーディングが使用されている場合は明示する必要がある

```py
import pandas as pd

pd.read_csv('data/pandas/read_csv_sjis.csv', encoding='cp932')

pd.read_csv('data/pandas/read_csv_sjis.csv', encoding='shift_jis')
```

<a id="markdown-圧縮された-csv-ファイルを読み込む" name="圧縮された-csv-ファイルを読み込む"></a>

##### 圧縮された CSV ファイルを読み込む

```py
import pandas as pd

# 拡張子が .gz / .bz2 / .zip / .xz でない場合は、
# 引数 compression に gz / bz2 / zip / xz 指定する
df1 = pd.read_csv('data/pandas/read_csv.zip')
print(df1)
```

```
   11  12  13  14
0  21  22  23  24
1  31  32  33  34
```

<a id="markdown-web-上の-csv-ファイルを読み込む" name="web-上の-csv-ファイルを読み込む"></a>

##### Web 上の CSV ファイルを読み込む

```py
import pandas as pd

df1 = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
print(df1)
```

```
     SepalLength  SepalWidth  PetalLength  PetalWidth            Name
0            5.1         3.5          1.4         0.2     Iris-setosa
1            4.9         3.0          1.4         0.2     Iris-setosa
2            4.7         3.2          1.3         0.2     Iris-setosa
3            4.6         3.1          1.5         0.2     Iris-setosa
4            5.0         3.6          1.4         0.2     Iris-setosa
..           ...         ...          ...         ...             ...
145          6.7         3.0          5.2         2.3  Iris-virginica
146          6.3         2.5          5.0         1.9  Iris-virginica
147          6.5         3.0          5.2         2.0  Iris-virginica
148          6.2         3.4          5.4         2.3  Iris-virginica
149          5.9         3.0          5.1         1.8  Iris-virginica

[150 rows x 5 columns]
```

<a id="markdown-時系列データ" name="時系列データ"></a>

##### 時系列データ

```py
import pandas as pd

df = pd.read_csv("data/pandas/time_series.csv", index_col=["Date"], parse_dates=True)
```

```
                 Amount
Date
2020-06-01  20200601188
2020-06-02  20200602528
2020-06-03  20200603881
2020-06-04  20200604539
2020-06-05  20200605607
...                 ...
2021-05-27  20210527352
2021-05-28  20210528819
2021-05-29  20210529704
2021-05-30  20210530767
2021-05-31  20210531754

[365 rows x 1 columns]
```

<a id="markdown-書き出し" name="書き出し"></a>

#### 書き出し

```py
import pandas as pd

df1 = pd.read_csv('data/pandas/read_csv.csv')

# CSVファイルに書き出し
df1.to_csv('data/pandas/to_csv.csv')

# 一部の列だけ書き出し
df1.to_csv('data/pandas/to_csv.csv', columns=['11'])

# 列名、行名の出力有無を指定
df1.to_csv('data/pandas/to_csv.csv', header=False, index=False)

# 文字コードを指定
df1.to_csv('data/pandas/to_csv.csv', encoding='cp932')

# 区切り文字を変更
df1.to_csv('data/pandas/to_csv.tsv', sep='\t')

# 既存のCSVファイルに追記
df1.to_csv('data/pandas/to_csv.csv')
df1.to_csv('data/pandas/to_csv.csv', mode='a', header=False)






```

<a id="markdown-tsv" name="tsv"></a>

### TSV

<a id="markdown-読み込み-1" name="読み込み-1"></a>

#### 読み込み

```py
import pandas as pd

df1 = pd.read_table('data/pandas/read_tsv.tsv')
print(df1)

df2 = pd.read_csv('data/pandas/read_tsv.tsv', sep='\t')
print(df2)
```

```
   11  12  13  14
0  21  22  23  24
1  31  32  33  34

   11  12  13  14
0  21  22  23  24
1  31  32  33  34
```

<a id="markdown-書き出し-1" name="書き出し-1"></a>

#### 書き出し

```py
import pandas as pd

df1 = pd.read_table('data/pandas/read_tsv.tsv')
df1.to_csv('data/pandas/to_csv.tsv', sep='\t')
```

<a id="markdown-json" name="json"></a>

### JSON

<a id="markdown-読み込み-2" name="読み込み-2"></a>

#### 読み込み

```py
import pandas as pd

df1 = pd.read_json('data/pandas/read_json.json')
print(df1)
```

```
      col1  col2
row1  val1  val4
row2  val2  val5
row3  val3    値6
```

<a id="markdown-書き出し-2" name="書き出し-2"></a>

#### 書き出し

```py
import pandas as pd

df1 = pd.read_json('data/pandas/read_json.json')
df1.to_json('data/pandas/to_json.json')
```

<a id="markdown-フォーマット" name="フォーマット"></a>

#### フォーマット

`orient` に、以下のいずれかを指定

- `'split'` : dict like `{'index' -> [index], 'columns' -> [columns], 'data' -> [values]}`
- `'records'` : list like `[{column -> value}, … , {column -> value}]`
- `'index'` : dict like `{index -> {column -> value}}`
- `'columns'` : dict like `{column -> {index -> value}}`
- `'values'` : just the values array
- `'table'` : dict like `{'schema': {schema}, 'data': {data}}`

```py
from pprint import pprint
import json
import pandas as pd

df = pd.DataFrame(
    [['val1', 'val2'], ['val3', 'val4']],
    index=['row1', 'row2'],
    columns=['col1', 'col2'])

pprint(
    json.loads(df.to_json(orient='split'))
    )

pprint(
    json.loads(df.to_json(orient='records'))
    )

pprint(
    json.loads(df.to_json(orient='index'))
    )

pprint(
    json.loads(df.to_json(orient='columns'))
    )

pprint(
    json.loads(df.to_json(orient='values'))
    )

pprint(
    json.loads(df.to_json(orient='table'))
    )
```

```
{'columns': ['col1', 'col2'],
 'data': [['val1', 'val2'], ['val3', 'val4']],
 'index': ['row1', 'row2']}

[{'col1': 'val1', 'col2': 'val2'}, {'col1': 'val3', 'col2': 'val4'}]

{'row1': {'col1': 'val1', 'col2': 'val2'},
 'row2': {'col1': 'val3', 'col2': 'val4'}}

{'col1': {'row1': 'val1', 'row2': 'val3'},
 'col2': {'row1': 'val2', 'row2': 'val4'}}

[['val1', 'val2'], ['val3', 'val4']]

{'data': [{'col1': 'val1', 'col2': 'val2', 'index': 'row1'},
          {'col1': 'val3', 'col2': 'val4', 'index': 'row2'}],
 'schema': {'fields': [{'name': 'index', 'type': 'string'},
                       {'name': 'col1', 'type': 'string'},
                       {'name': 'col2', 'type': 'string'}],
            'pandas_version': '0.20.0',
            'primaryKey': ['index']}}
```

<a id="markdown-json-文字列の読み込み" name="json-文字列の読み込み"></a>

### JSON 文字列の読み込み

```py
import pandas as pd

json_str = '{"col1":{"row1":"val1","row2":"val2","row3":"val3"},"col2":{"row1":"val4","row2":"val5","row3":"\u50246"}}'

json_str = json_str.replace("'", '"')
df1 = pd.read_json(json_str)
print(df1)
```

```
   col1  col2
0  val1  val2
1  val3  val4
2  val5    値6
```

<a id="markdown-json-lines" name="json-lines"></a>

### JSON Lines

```py
import pandas as pd

df1 = pd.read_json('data/pandas/read_json.jsonl', orient='records', lines=True)
print(df1)
```

```
   col1  col2
0  val1  val2
1  val3  val4
2  val5    値6
```

<a id="markdown-excel-形式xlsx" name="excel-形式xlsx"></a>

### Excel 形式（xlsx）

<a id="markdown-依存パッケージをインストール" name="依存パッケージをインストール"></a>

#### 依存パッケージをインストール

- 読み込み

```ps
$ pip install xlrd
```

- 書き込み

```ps
$ pip install xlwt
$ pip install openpyxl
```

<a id="markdown-読み込み-3" name="読み込み-3"></a>

#### 読み込み

```py
import pandas as pd

# Excel形式のファイルから読み込む
df1 = pd.read_excel('data/pandas/read_excel.xlsx')

# 読み込むシートを指定する
df1 = pd.read_excel('data/pandas/read_excel.xlsx') # 指定しないと最初のシートが読み込まれる
df1 = pd.read_excel('data/src/sample.xlsx', sheet_name=0) # 0番目のシート
df1 = pd.read_excel('data/src/sample.xlsx', sheet_name='Sheet1') # シート名を指定
df2 = pd.read_excel('data/src/sample.xlsx', sheet_name=[0, 'Sheet2']) # 複数のシートを指定
print(df2[0]) # シートのインデックスを指定して表示
print(df2['Sheet2']) # シート名を指定して表示
df3 = pd.read_excel('data/src/sample.xlsx', sheet_name=None) # 全てのシートを読み込み

# 列名、行名として扱う行、列を指定
df4 = pd.read_excel('data/src/sample.xlsx', header=None, index_col=None) # インデックス番号を使用
df5 = pd.read_excel('data/src/sample.xlsx', header=0)
df6 = pd.read_excel('data/src/sample.xlsx', index_col=0)

# 行と列を抽出
df7 = pd.read_excel('data/src/sample.xlsx', usecols=[0, 1])
df8 = pd.read_excel('data/src/sample.xlsx', skiprows=[1])
df9 = pd.read_excel('data/src/sample.xlsx', skipfooter=1)
```

<a id="markdown-書き出し-3" name="書き出し-3"></a>

#### 書き出し

```py
import pandas as pd

# 書き出し
df1 = pd.read_excel('data/pandas/read_excel.xlsx')
df1.to_excel('data/pandas/to_excel.xlsx')

# 複数のDataFrameを1つのExcel形式ファイルに書き出し
with pd.ExcelWriter('data/pandas/to_excel.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df1.to_excel(writer, sheet_name='Sheet2')

# 既存ファイルにシートを追加
with pd.ExcelWriter('data/pandas/to_excel.xlsx') as writer:
    writer.book = openpyxl.load_workbook(path)
    df1.to_excel(writer, sheet_name='Sheet3')
```

<a id="markdown-pickle" name="pickle"></a>

### pickle

<a id="markdown-読み込み-4" name="読み込み-4"></a>

#### 読み込み

```py
import pandas as pd

df1 = pd.read_pickle('data/pandas/read_pickle.pkl')
print(df1)
```

<a id="markdown-書き出し-4" name="書き出し-4"></a>

#### 書き出し

```py
import pandas as pd

df1 = pd.read_pickle('data/pandas/read_pickle.pkl')
df1.to_pickle('data/pandas/to_pickle.pkl')
```

<a id="markdown-html" name="html"></a>

### HTML

<a id="markdown-web-ページ内の-table表を読み込む" name="web-ページ内の-table表を読み込む"></a>

#### Web ページ内の table（表）を読み込む

<a id="markdown-依存パッケージをインストール-1" name="依存パッケージをインストール-1"></a>

##### 依存パッケージをインストール

```ps
$ pip install beautifulsoup4 html5lib lxml
```

<a id="markdown-読み込み-5" name="読み込み-5"></a>

##### 読み込み

```py
from pprint import pprint
import pandas as pd

url = 'https://indexes.nikkei.co.jp/nkave/archives/data'
dfs = pd.read_html(url)
for df in dfs:
    pprint(df)

dfs = pd.read_html(
    url,
    # attrs = {'id': 'table'},
    # encoding='cp932'
    header=0,
    index_col=0,
    # skiprows=1,
    match='.+値'
    )

for df in dfs:
    pprint(df[['始値', '終値']].head())
```

```
                  始値        終値
日付
2020.05.01  19991.97  19619.35
2020.05.07  19468.52  19674.77
2020.05.08  19972.09  20179.09
2020.05.11  20333.73  20390.66
2020.05.12  20413.23  20366.48
```

<a id="markdown-データ抽出" name="データ抽出"></a>

## データ抽出

<a id="markdown-dataframe-の概要を確認" name="dataframe-の概要を確認"></a>

### Dataframe の概要を確認

<a id="markdown-情報表示・統計量" name="情報表示・統計量"></a>

#### 情報表示・統計量

```py
import numpy as np
import pandas as pd

arr = np.arange(50).reshape(5, 10)
idx = pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index')
col = pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column')
df = pd.DataFrame(arr, index=idx, columns=col) # 2次元: データフレーム

df.info()

# 統計量
df.describe()
# df.describe(include='all') # 数値型以外も対象とする
# df.describe(include=['object', int]) # object型とint型のみ対象とする

df.astype('str').describe() ## 数値型をカテゴリカルデータとして扱う

df.astype({'c1': int, 'c2': int}).describe() ## 数値文字列を数値型として扱う

# 要素数
df.count()
df['c1'].count()

# 一意な要素の数
df.nunique()
df['c1'].nunique()

# 頻度
df['c1'].value_counts()

# 最頻値（最頻値が複数ある時は全部返す）
df.mode()
df['c1'].mode()
df.mode().count() ## 最頻値が複数ある時の個数

# 算術平均
df.mean(numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].mean()

# 標準偏差
df.std(numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].std()

# 最小値
df.min(numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].min()

# 最大値
df.max(numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].max()

# 中央値
df.median(numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].median()

# 分位数（第1四分位数、第3四分位数）
df.quantile(q=[0.25, 0.75], numeric_only=True) # Falseの場合、bool型の列のtrueは1、falseは0と見なされる
df['c1'].quantile(q=[0.25, 0.75])
```

```
# info
<class 'pandas.core.frame.DataFrame'>
Index: 5 entries, r1 to r5
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   c1      5 non-null      int32
 1   c2      5 non-null      int32
 2   c3      5 non-null      int32
 3   c4      5 non-null      int32
 4   c5      5 non-null      int32
 5   c6      5 non-null      int32
 6   c7      5 non-null      int32
 7   c8      5 non-null      int32
 8   c9      5 non-null      int32
 9   c10     5 non-null      int32
dtypes: int32(10)
memory usage: 240.0+ bytes


# 統計量
column         c1         c2         c3         c4         c5         c6         c7         c8         c9        c10
count    5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000
mean    20.000000  21.000000  22.000000  23.000000  24.000000  25.000000  26.000000  27.000000  28.000000  29.000000
std     15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388
min      0.000000   1.000000   2.000000   3.000000   4.000000   5.000000   6.000000   7.000000   8.000000   9.000000
25%     10.000000  11.000000  12.000000  13.000000  14.000000  15.000000  16.000000  17.000000  18.000000  19.000000
50%     20.000000  21.000000  22.000000  23.000000  24.000000  25.000000  26.000000  27.000000  28.000000  29.000000
75%     30.000000  31.000000  32.000000  33.000000  34.000000  35.000000  36.000000  37.000000  38.000000  39.000000
max     40.000000  41.000000  42.000000  43.000000  44.000000  45.000000  46.000000  47.000000  48.000000  49.000000


## 数値型をカテゴリカルデータとして扱う
column  c1  c2  c3  c4  c5  c6 c7  c8  c9 c10
count    5   5   5   5   5   5  5   5   5   5
unique   5   5   5   5   5   5  5   5   5   5
top     20  31  42  23  14  15  6  37  18  39
freq     1   1   1   1   1   1  1   1   1   1

## 数値文字列を数値型として扱う
column         c1         c2         c3         c4         c5         c6         c7         c8         c9        c10
count    5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000   5.000000
mean    20.000000  21.000000  22.000000  23.000000  24.000000  25.000000  26.000000  27.000000  28.000000  29.000000
std     15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388  15.811388
min      0.000000   1.000000   2.000000   3.000000   4.000000   5.000000   6.000000   7.000000   8.000000   9.000000
25%     10.000000  11.000000  12.000000  13.000000  14.000000  15.000000  16.000000  17.000000  18.000000  19.000000
50%     20.000000  21.000000  22.000000  23.000000  24.000000  25.000000  26.000000  27.000000  28.000000  29.000000
75%     30.000000  31.000000  32.000000  33.000000  34.000000  35.000000  36.000000  37.000000  38.000000  39.000000
max     40.000000  41.000000  42.000000  43.000000  44.000000  45.000000  46.000000  47.000000  48.000000  49.000000


# 要素数
column
c1     5
c2     5
c3     5
c4     5
c5     5
c6     5
c7     5
c8     5
c9     5
c10    5
dtype: int64

5


# 一意な要素の数
column
c1     5
c2     5
c3     5
c4     5
c5     5
c6     5
c7     5
c8     5
c9     5
c10    5
dtype: int64

5


# 頻度
30    1
40    1
20    1
10    1
0     1
Name: c1, dtype: int64


# 最頻値（最頻値が複数ある時は全部返す）
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
0        0   1   2   3   4   5   6   7   8    9
1       10  11  12  13  14  15  16  17  18   19
2       20  21  22  23  24  25  26  27  28   29
3       30  31  32  33  34  35  36  37  38   39
4       40  41  42  43  44  45  46  47  48   49

0     0
1    10
2    20
3    30
4    40
dtype: int32

## 最頻値が複数ある時の個数
column
c1     5
c2     5
c3     5
c4     5
c5     5
c6     5
c7     5
c8     5
c9     5
c10    5
dtype: int64


# 算術平均
column
c1     20.0
c2     21.0
c3     22.0
c4     23.0
c5     24.0
c6     25.0
c7     26.0
c8     27.0
c9     28.0
c10    29.0
dtype: float64

20.0


# 標準偏差
column
c1     15.811388
c2     15.811388
c3     15.811388
c4     15.811388
c5     15.811388
c6     15.811388
c7     15.811388
c8     15.811388
c9     15.811388
c10    15.811388
dtype: float64

15.811388300841896

# 最小値
column
c1     0
c2     1
c3     2
c4     3
c5     4
c6     5
c7     6
c8     7
c9     8
c10    9
dtype: int64

0

# 最大値
column
c1     40
c2     41
c3     42
c4     43
c5     44
c6     45
c7     46
c8     47
c9     48
c10    49
dtype: int64

40


# 中央値
column
c1     20.0
c2     21.0
c3     22.0
c4     23.0
c5     24.0
c6     25.0
c7     26.0
c8     27.0
c9     28.0
c10    29.0
dtype: float64

20.0


# 分位数（第1四分位数、第3四分位数）
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
0.25    10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
0.75    30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0

0.25    10.0
0.75    30.0
Name: c1, dtype: float64





```

<a id="markdown-行名・列名行数・列数・要素数" name="行名・列名行数・列数・要素数"></a>

#### 行名・列名、行数・列数・要素数

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

df.index # 行名
df.columns # 列名
df.values # 値
```

```
# 行名
Index(['r1', 'r2', 'r3', 'r4', 'r5'], dtype='object', name='index')

# 列名
Index(['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'], dtype='object', name='column')

# 値
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
```

<a id="markdown-行名を変更" name="行名を変更"></a>

##### 行名を変更

<a id="markdown-列をインデックスとして使用" name="列をインデックスとして使用"></a>

###### 列をインデックスとして使用

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df.index = df.pop('c2')

print(df)
print(df.index) # 行名
```

```
column  c1  c3  c4  c5  c6  c7  c8  c9  c10
c2
1        0   2   3   4   5   6   7   8    9
11      10  12  13  14  15  16  17  18   19
21      20  22  23  24  25  26  27  28   29
31      30  32  33  34  35  36  37  38   39
41      40  42  43  44  45  46  47  48   49

Int64Index([1, 11, 21, 31, 41], dtype='int64', name='c2')
```

<a id="markdown-連番にリセットする" name="連番にリセットする"></a>

###### 連番にリセットする

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

# 変更前
print(df)
print(df.index) # 行名

# 変更後
df = df.reset_index(drop=True)
print(df)
print(df.index) # 行名
```

```
# 変更前
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

Index(['r1', 'r2', 'r3', 'r4', 'r5'], dtype='object', name='index')

# 変更後
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
0        0   1   2   3   4   5   6   7   8    9
1       10  11  12  13  14  15  16  17  18   19
2       20  21  22  23  24  25  26  27  28   29
3       30  31  32  33  34  35  36  37  38   39
4       40  41  42  43  44  45  46  47  48   49

RangeIndex(start=0, stop=5, step=1)
```

<a id="markdown-行名・列名行数・列数・要素数-1" name="行名・列名行数・列数・要素数-1"></a>

#### 行名・列名、行数・列数・要素数

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

print(len(df)) # 行数
print(len(df.columns)) # 列数
print(df.shape) # タプル (行数, 列数)
r, c = df.shape
print(r, c)

print(df.size) # 要素数

print(df.nunique()) # 重複しない要素の数
```

```
# 行数
5

# 列数
10

# タプル (行数, 列数)
(5, 10)
5 10

# 要素数
50

# 重複しない要素の数
column
c1     5
c2     5
c3     5
c4     5
c5     5
c6     5
c7     5
c8     5
c9     5
c10    5
dtype: int64
```

<a id="markdown-欠損値の数" name="欠損値の数"></a>

#### 欠損値の数

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

print(df.isnull().sum()) # 各列ごとのNaNの数
print(df.isnull().sum(axis=1)) # 各行ごとのNaNの数

print(df.count()) # 各列ごとのNaNでない要素の数
print(df.count(axis=1)) # 各行ごとのNaNでない要素の数
```

```
# 各列ごとのNaNの数
column
c1     0
c2     0
c3     0
c4     0
c5     0
c6     0
c7     0
c8     0
c9     0
c10    0
dtype: int64

# 各行ごとのNaNの数
index
r1    0
r2    0
r3    0
r4    0
r5    0
dtype: int64

# 各列ごとのNaNでない要素の数
column
c1     5
c2     5
c3     5
c4     5
c5     5
c6     5
c7     5
c8     5
c9     5
c10    5
dtype: int64

# 各行ごとのNaNでない要素の数
index
r1    10
r2    10
r3    10
r4    10
r5    10
dtype: int64
```

<a id="markdown-行を絞る" name="行を絞る"></a>

#### 行を絞る

<a id="markdown-先頭" name="先頭"></a>

##### 先頭

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(100).reshape(10, 10), index=pd.Index(['r{}'.format(x+1) for x in range(10)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

df.head() # 先頭5行
df.head(2) # 先頭2行
```

```
# 先頭5行
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

# 先頭2行
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
```

<a id="markdown-末尾" name="末尾"></a>

##### 末尾

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(100).reshape(10, 10), index=pd.Index(['r{}'.format(x+1) for x in range(10)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

df.head() # 先頭5行
df.head(2) # 先頭2行
```

```
# 末尾5行
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r6      50  51  52  53  54  55  56  57  58   59
r7      60  61  62  63  64  65  66  67  68   69
r8      70  71  72  73  74  75  76  77  78   79
r9      80  81  82  83  84  85  86  87  88   89
r10     90  91  92  93  94  95  96  97  98   99

# 末尾2行
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r9      80  81  82  83  84  85  86  87  88   89
r10     90  91  92  93  94  95  96  97  98   99
```

<a id="markdown-インデックスから行と列を絞るdataframe" name="インデックスから行と列を絞るdataframe"></a>

#### インデックスから行と列を絞る（DataFrame）

<a id="markdown-行または列" name="行または列"></a>

##### 行または列

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

df['c1'] # df[列名]
df.c1 # df.列名

df[['c1']]       # df[列名リスト]
df[['c1', 'c9']] #

# df[行番号スライス]
df[:1]  # r1
df[2:3] # r3
df[4:6] # r5, r6
df[7:]  # r8からr20
```

```
# df['c1'] # df[列名]
index
r1       0
r2      10
r3      20
r4      30
r5      40
r6      50
r7      60
r8      70
r9      80
r10     90
r11    100
r12    110
r13    120
r14    130
r15    140
r16    150
r17    160
r18    170
r19    180
r20    190
Name: c1, dtype: int32

# df['c1'] # df[列名]
index
r1       0
r2      10
r3      20
r4      30
r5      40
r6      50
r7      60
r8      70
r9      80
r10     90
r11    100
r12    110
r13    120
r14    130
r15    140
r16    150
r17    160
r18    170
r19    180
r20    190
Name: c1, dtype: int32

# df[['c1', 'c9']]
column   c1   c9
index
r1        0    8
r2       10   18
r3       20   28
r4       30   38
r5       40   48
r6       50   58
r7       60   68
r8       70   78
r9       80   88
r10      90   98
r11     100  108
r12     110  118
r13     120  128
r14     130  138
r15     140  148
r16     150  158
r17     160  168
r18     170  178
r19     180  188
r20     190  198

# df[行番号スライス]
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9

# df[2:3]
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r3      20  21  22  23  24  25  26  27  28   29

# df[4:6]
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r5      40  41  42  43  44  45  46  47  48   49
r6      50  51  52  53  54  55  56  57  58   59

# df[7:]
column   c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
index
r8       70   71   72   73   74   75   76   77   78   79
r9       80   81   82   83   84   85   86   87   88   89
r10      90   91   92   93   94   95   96   97   98   99
r11     100  101  102  103  104  105  106  107  108  109
r12     110  111  112  113  114  115  116  117  118  119
r13     120  121  122  123  124  125  126  127  128  129
r14     130  131  132  133  134  135  136  137  138  139
r15     140  141  142  143  144  145  146  147  148  149
r16     150  151  152  153  154  155  156  157  158  159
r17     160  161  162  163  164  165  166  167  168  169
r18     170  171  172  173  174  175  176  177  178  179
r19     180  181  182  183  184  185  186  187  188  189
r20     190  191  192  193  194  195  196  197  198  199
```

<a id="markdown-行と列" name="行と列"></a>

##### 行と列

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))


# 特定の要素にアクセス

# 行名と列名の組み合わせで指定
print(df.at['r16', 'c5']) # 値を取得 154
df.at['r16', 'c5'] = 2 * df.at['r16', 'c5'] # 上書き

# 行番号と列番号の組み合わせで指定
print(df.iat[15, 4]) # 値を取得 308
df.iat[15, 4] = 3 * df.iat[15, 4] # 上書き 924

# 複数の要素にアクセス

# 行全体・列全体
print(df.loc['r16']) # 引数1つだけ指定すると行指定
print(df.loc[:, 'c5']) # 列だけ指定

# 行名と列名の組み合わせで指定
print(df.loc['r16', 'c5'])
print(df.loc['r15':'r17', 'c4':'c6'])
print(df.loc[['r15', 'r17'], ['c4', 'c6']])

# 行番号と列番号の組み合わせで指定
print(df.iloc[15, 4])
print(df.iloc[14:16, 3:5])
print(df.loc[['r15', 'r17'], ['c4', 'c6']])

print(df.iloc[::2, 1::2]) # 偶奇

# 名前と番号の組み合わせで指定
print(df.at[df.index[15], 'c5'])
print(df.loc[['r15', 'r17'], df.columns[4]])
```

```
# 行名と列名の組み合わせで指定
# 行名と列名の組み合わせで指定
924

column   c4   c5   c6
index
r15     143  144  145
r16     153  924  155
r17     163  164  165

column   c4   c6
index
r15     143  145
r17     163  165

# 行番号と列番号の組み合わせで指定
924

column   c4   c5
index
r15     143  144
r16     153  924

column   c4   c6
index
r15     143  145
r17     163  165

column   c2   c4   c6   c8  c10
index
r1        1    3    5    7    9
r3       21   23   25   27   29
r5       41   43   45   47   49
r7       61   63   65   67   69
r9       81   83   85   87   89
r11     101  103  105  107  109
r13     121  123  125  127  129
r15     141  143  145  147  149
r17     161  163  165  167  169
r19     181  183  185  187  189

# 名前と番号の組み合わせで指定
924

index
r15    144
r17    164
Name: c5, dtype: int32
```

<a id="markdown-添え字で辿っていく方法" name="添え字で辿っていく方法"></a>

###### 添え字で辿っていく方法

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

print(df['c3']['r9'])

print(df[['c2','c4']]['r8':'r10'])
# print(df[['c2','c4']][['r8','r10']]) # KeyError

print(df[1:3])
```

```
# print(df['c3']['r9'])
82

# print(df[['c2','c4']]['r8':'r10'])
column  c2  c4
index
r8      71  73
r9      81  83
r10     91  93
```

<a id="markdown-インデックスから行を絞るseries" name="インデックスから行を絞るseries"></a>

#### インデックスから行を絞る（Series）

```py
import numpy as np
import pandas as pd

sr = pd.Series(np.arange(10), index=pd.Index(['r{}'.format(x+1) for x in range(10)], name = 'index'))

sr['r2']         # sr[行名]       → 1 <class 'numpy.int32'>
sr[['r5', 'r6']] # sr[行名リスト]
sr['r2':'r4']    # sr[行名スライス] → r2, r3, r4

sr[9]            # sr[行番号]         → (r10) 9 <class 'numpy.int32'>
sr[[1, 2, 3]]    # sr[行番号リスト]   → r2, r3, r4
sr[1:3]          # sr[行番号スライス] → r2, r3
```

```
# sr['r2']
1

# sr[['r5', 'r6']]
index
r5    4
r6    5
dtype: int32

# sr['r2':'r4']
index
r2    1
r3    2
r4    3
dtype: int32

# sr[9]
9

# sr[[1, 2, 3]]
index
r2    1
r3    2
r4    3
dtype: int32

# sr[1:3]
index
r2    1
r3    2
dtype: int32
```

<a id="markdown-条件に適合する行を抽出" name="条件に適合する行を抽出"></a>

#### 条件に適合する行を抽出

<a id="markdown-ブールインデックス" name="ブールインデックス"></a>

##### ブールインデックス

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })

df[[True, True, True, False, True]]

df[df['price'] < 456]
df[(df['price'] <= 456) | (df['item'] == 'bar')] # OR
df[~(df['item'] == 'piyo') & (df['item'] == 'piyo')] # NOT, AND
```

```
         date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
4  2020/05/04  fuga     56

         date  item  price
4  2020/05/04  fuga     56

         date  item  price
1  2020/05/01   bar  23456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

Empty DataFrame
Columns: [date, item, price]
Index: []
```

<a id="markdown-query-メソッド" name="query-メソッド"></a>

##### query メソッド

- 事前準備（必須ではない）

```ps
$ pip install numexpr
```

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })

max_index = 3
df.query('index < @max_index') # 変数を参照して、インデックス番号と比較

df.query('price < 456')

df.query('price <= 456 or item == "bar"') # OR
df.query('price <= 456 | item == "bar"')

df.query('not item == "2020/05/01" and item == "2020/05/01"') # NOT, AND

df.query('12345 <= price < 20000') # ANDを使用しない範囲指定
```

```
   item  price  cost
0   foo  12345  4321
1   bar  23456  5432
2  hoge   3456   654

         date  item  price
4  2020/05/04  fuga     56

         date  item  price
1  2020/05/01   bar  23456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

         date  item  price
1  2020/05/01   bar  23456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

Empty DataFrame
Columns: [date, item, price]
Index: []

         date item  price
0  2020/05/01  foo  12345
```

```py
import pandas as pd

df = pd.DataFrame({
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56],
    'cost': [4321, 5432, 654, 76, 87]
    })

df.query('price < 3 * cost') # 他の列を参照

df.query('item in ["foo", "hoge"]') # in演算子
df.query('item == ["foo", "hoge"]')

print(df.query('item.str.startswith("f")')) # 前方一致
print(df.query('item.str.endswith("o")'))   # 後方一致
print(df.query('item.str.contains("oo")'))  # 部分一致
print(df.query('item.str.match("[a-c]")'))  # 正規表現

df.query('price.astype("str").str.endswith("6")') # 文字型以外の列
```

```
# 他の列を参照
   item  price  cost
0   foo  12345  4321
4  fuga     56    87

# in演算子
   item  price  cost
0   foo  12345  4321
2  hoge   3456   654

# 前方一致
   item  price  cost
0   foo  12345  4321
4  fuga     56    87

# 後方一致
   item  price  cost
0   foo  12345  4321
3  piyo    456    76

# 部分一致
  item  price  cost
0  foo  12345  4321

# 正規表現
  item  price  cost
1  bar  23456  5432

# 文字型以外の列
   item  price  cost
1   bar  23456  5432
2  hoge   3456   654
3  piyo    456    76
4  fuga     56    87
```

<a id="markdown-条件に適合する列を抽出" name="条件に適合する列を抽出"></a>

#### 条件に適合する列を抽出

<a id="markdown-ブールインデックス-1" name="ブールインデックス-1"></a>

##### ブールインデックス

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })

print(df.loc[:, df.columns.str.endswith('e')])
print(df.loc[:, df.columns.str.endswith('m') | df.columns.str.endswith('e')])
```

```
         date  price
0  2020/05/01  12345
1  2020/05/01  23456
2  2020/05/02   3456
3  2020/05/03    456
4  2020/05/04     56

         date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56
```

<a id="markdown-欠損値を除去" name="欠損値を除去"></a>

#### 欠損値を除去

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df = df.replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.nan)
print(df)

df.dropna(how='all') # 欠損値しかない行を除去
```

```
# df
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
r2       NaN  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# df.dropna(how='all')
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r2       NaN  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0
```

<a id="markdown-ソート" name="ソート"></a>

#### ソート

<a id="markdown-行や列の値でソート" name="行や列の値でソート"></a>

##### 行や列の値でソート

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df.head()

# df_random1 = df.sort_values('c1', ascending=False)
df_random1 = df.sort_values(['c1', 'c2'], ascending=[False, False])
df_random1.head()

```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

column   c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
index
r20     190  191  192  193  194  195  196  197  198  199
r19     180  181  182  183  184  185  186  187  188  189
r18     170  171  172  173  174  175  176  177  178  179
r17     160  161  162  163  164  165  166  167  168  169
r16     150  151  152  153  154  155  156  157  158  159
```

<a id="markdown-行や列のインデックスでソート" name="行や列のインデックスでソート"></a>

##### 行や列のインデックスでソート

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df.head()

df_random1 = df.sort_index(axis=0, ascending=False)
df_random1.head()

df_random2 = df.sort_index(axis=1, ascending=False)
df_random2.head()

```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r9      80  81  82  83  84  85  86  87  88   89
r8      70  71  72  73  74  75  76  77  78   79
r7      60  61  62  63  64  65  66  67  68   69
r6      50  51  52  53  54  55  56  57  58   59
r5      40  41  42  43  44  45  46  47  48   49

column  c9  c8  c7  c6  c5  c4  c3  c2  c10  c1
index
r1       8   7   6   5   4   3   2   1    9   0
r2      18  17  16  15  14  13  12  11   19  10
r3      28  27  26  25  24  23  22  21   29  20
r4      38  37  36  35  34  33  32  31   39  30
r5      48  47  46  45  44  43  42  41   49  40
```

<a id="markdown-ランダムソート" name="ランダムソート"></a>

##### ランダムソート

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(200).reshape(20, 10), index=pd.Index(['r{}'.format(x+1) for x in range(20)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df.head()

# df.sample(frac=1, random_state=12345) # 乱数の種（シード）を固定する
df_random1 = df.sample(frac=1)
df_random1.head()

# indexをリセットする
df_random2 = df_random1.reset_index(drop=True)
print(df_random2)

# indexに代入する
df_random2.index = ['r{}'.format(x+1) for x in range(20)]
print(df_random2)
```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

column   c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
index
r8       70   71   72   73   74   75   76   77   78   79
r20     190  191  192  193  194  195  196  197  198  199
r15     140  141  142  143  144  145  146  147  148  149
r2       10   11   12   13   14   15   16   17   18   19
r5       40   41   42   43   44   45   46   47   48   49

column   c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
0        10   11   12   13   14   15   16   17   18   19
1        60   61   62   63   64   65   66   67   68   69
2        50   51   52   53   54   55   56   57   58   59
3        80   81   82   83   84   85   86   87   88   89
4       180  181  182  183  184  185  186  187  188  189
5       110  111  112  113  114  115  116  117  118  119
6       160  161  162  163  164  165  166  167  168  169
7       150  151  152  153  154  155  156  157  158  159
8       120  121  122  123  124  125  126  127  128  129
9       100  101  102  103  104  105  106  107  108  109
10       70   71   72   73   74   75   76   77   78   79
11      190  191  192  193  194  195  196  197  198  199
12      140  141  142  143  144  145  146  147  148  149
13       30   31   32   33   34   35   36   37   38   39
14      130  131  132  133  134  135  136  137  138  139
15       20   21   22   23   24   25   26   27   28   29
16        0    1    2    3    4    5    6    7    8    9
17       90   91   92   93   94   95   96   97   98   99
18      170  171  172  173  174  175  176  177  178  179
19       40   41   42   43   44   45   46   47   48   49

column   c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
r1       10   11   12   13   14   15   16   17   18   19
r2       60   61   62   63   64   65   66   67   68   69
r3       50   51   52   53   54   55   56   57   58   59
r4       80   81   82   83   84   85   86   87   88   89
r5      180  181  182  183  184  185  186  187  188  189
r6      110  111  112  113  114  115  116  117  118  119
r7      160  161  162  163  164  165  166  167  168  169
r8      150  151  152  153  154  155  156  157  158  159
r9      120  121  122  123  124  125  126  127  128  129
r10     100  101  102  103  104  105  106  107  108  109
r11      70   71   72   73   74   75   76   77   78   79
r12     190  191  192  193  194  195  196  197  198  199
r13     140  141  142  143  144  145  146  147  148  149
r14      30   31   32   33   34   35   36   37   38   39
r15     130  131  132  133  134  135  136  137  138  139
r16      20   21   22   23   24   25   26   27   28   29
r17       0    1    2    3    4    5    6    7    8    9
r18      90   91   92   93   94   95   96   97   98   99
r19     170  171  172  173  174  175  176  177  178  179
r20      40   41   42   43   44   45   46   47   48   49
```

<a id="markdown-データ加工-1" name="データ加工-1"></a>

## データ加工

<a id="markdown-追加" name="追加"></a>

### 追加

<a id="markdown-行の追加" name="行の追加"></a>

#### 行の追加

<a id="markdown-append" name="append"></a>

##### append

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })
df

df = df.append(
    pd.DataFrame(['2020/06/01','added',99999], index=df.columns).T
)
df

df = df.append(
    {'date': '2020/06/01', 'item': 'added2', 'price': 999999},
    ignore_index=True
)
df

df = df.append(df)
df
```

```
         date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

         date   item  price
0  2020/05/01    foo  12345
1  2020/05/01    bar  23456
2  2020/05/02   hoge   3456
3  2020/05/03   piyo    456
4  2020/05/04   fuga     56
0  2020/06/01  added  99999

         date    item   price
0  2020/05/01     foo   12345
1  2020/05/01     bar   23456
2  2020/05/02    hoge    3456
3  2020/05/03    piyo     456
4  2020/05/04    fuga      56
5  2020/06/01   added   99999
6  2020/06/01  added2  999999

         date    item   price
0  2020/05/01     foo   12345
1  2020/05/01     bar   23456
2  2020/05/02    hoge    3456
3  2020/05/03    piyo     456
4  2020/05/04    fuga      56
5  2020/06/01   added   99999
6  2020/06/01  added2  999999
0  2020/05/01     foo   12345
1  2020/05/01     bar   23456
2  2020/05/02    hoge    3456
3  2020/05/03    piyo     456
4  2020/05/04    fuga      56
5  2020/06/01   added   99999
6  2020/06/01  added2  999999
```

<a id="markdown-concat" name="concat"></a>

##### concat

```py
import pandas as pd

df1 = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })
df2 = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'sale': [1234, 2345, 345, 45, 5]
    })

print('df1', df1)
print('df2', df2)

df3 = pd.concat([df1, df2])
print('df3', df3)

df4 = pd.concat(
    [df1, df2],
    join='inner', # 共通する列のみ残す
    ignore_index=True # インデックスを振り直す
)
print('df4', df4)
```

```
df1          date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

df2          date  item  sale
0  2020/05/01   foo  1234
1  2020/05/01   bar  2345
2  2020/05/02  hoge   345
3  2020/05/03  piyo    45
4  2020/05/04  fuga     5

df3          date  item    price    sale
0  2020/05/01   foo  12345.0     NaN
1  2020/05/01   bar  23456.0     NaN
2  2020/05/02  hoge   3456.0     NaN
3  2020/05/03  piyo    456.0     NaN
4  2020/05/04  fuga     56.0     NaN
0  2020/05/01   foo      NaN  1234.0
1  2020/05/01   bar      NaN  2345.0
2  2020/05/02  hoge      NaN   345.0
3  2020/05/03  piyo      NaN    45.0
4  2020/05/04  fuga      NaN     5.0

df4          date  item
0  2020/05/01   foo
1  2020/05/01   bar
2  2020/05/02  hoge
3  2020/05/03  piyo
4  2020/05/04  fuga
5  2020/05/01   foo
6  2020/05/01   bar
7  2020/05/02  hoge
8  2020/05/03  piyo
9  2020/05/04  fuga
```

<a id="markdown-列の追加" name="列の追加"></a>

#### 列の追加

<a id="markdown-列名" name="列名"></a>

##### 列名

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })
df

# 定数
df['amount'] = 0
df

# リスト（DataFrameの各行に要素を代入）
len(df)
df['number'] = range(len(df))
df

# 列の演算
df['tax'] = (0.1 * df['price']).round().astype(int)
```

```
         date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

         date  item  price  amount
0  2020/05/01   foo  12345       0
1  2020/05/01   bar  23456       0
2  2020/05/02  hoge   3456       0
3  2020/05/03  piyo    456       0
4  2020/05/04  fuga     56       0

         date  item  price  amount  number
0  2020/05/01   foo  12345       0       0
1  2020/05/01   bar  23456       0       1
2  2020/05/02  hoge   3456       0       2
3  2020/05/03  piyo    456       0       3
4  2020/05/04  fuga     56       0       4

         date  item  price  amount  number   tax
0  2020/05/01   foo  12345       0       0  1234
1  2020/05/01   bar  23456       0       1  2346
2  2020/05/02  hoge   3456       0       2   346
3  2020/05/03  piyo    456       0       3    46
4  2020/05/04  fuga     56       0       4     6
```

<a id="markdown-assign" name="assign"></a>

##### assign

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })
df

# 定数
df = df.assign(
    amount=0,
    number=range(len(df)),
    tax=(0.1 * df['price']).round().astype(int)
)
df
```

```
         date  item  price  amount  number   tax
0 2020/05/01 foo 12345 0 0 1234
1 2020/05/01 bar 23456 0 1 2346
2 2020/05/02 hoge 3456 0 2 346
3 2020/05/03 piyo 456 0 3 46
4 2020/05/04 fuga 56 0 4 6
```

<a id="markdown-insert位置を指定して追加" name="insert位置を指定して追加"></a>

##### insert（位置を指定して追加）

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
    'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
    'price': [12345, 23456, 3456, 456, 56]
    })
df

# 定数
df.insert(
    len(df.columns), # 挿入する位置を指定（ここでは末尾）
    'number', # 列名
    range(len(df)) # 値
)
df
```

```
         date  item  price
0  2020/05/01   foo  12345
1  2020/05/01   bar  23456
2  2020/05/02  hoge   3456
3  2020/05/03  piyo    456
4  2020/05/04  fuga     56

         date  item  price  number
0  2020/05/01   foo  12345       0
1  2020/05/01   bar  23456       1
2  2020/05/02  hoge   3456       2
3  2020/05/03  piyo    456       3
4  2020/05/04  fuga     56       4
```

<a id="markdown-concatdataframe-に-series-を列として追加" name="concatdataframe-に-series-を列として追加"></a>

##### concat（DataFrame に Series を列として追加）

```py
import pandas as pd

df = pd.DataFrame(
    {
        'date': ['2020/05/01', '2020/05/01', '2020/05/02', '2020/05/03', '2020/05/04'],
        'item': ['foo', 'bar', 'hoge', 'piyo', 'fuga'],
        'price': [12345, 23456, 3456, 456, 56]
    },
    index=['r{}'.format(x+1) for x in range(5)]
)
df

ss = pd.Series(range(1, 4, 1), index=['r{}'.format(x+2) for x in range(3)], name='number')
ss

# indexが同じレコードを連結する（存在しない場合は要素がNaN）
df1 = pd.concat([df, ss], axis=1)
print(df1)

# indexが同じレコードを連結する（存在しない場合はレコード自体なくなる）
df2 = pd.concat([df, ss], axis=1, join='inner')
print(df2)

# 3つ以上連結
df3 = pd.concat([df, df, ss, ss], axis=1, join='inner')
print(df3)
```

```
          date  item  price
r1  2020/05/01   foo  12345
r2  2020/05/01   bar  23456
r3  2020/05/02  hoge   3456
r4  2020/05/03  piyo    456
r5  2020/05/04  fuga     56

r2    1
r3    2
r4    3
Name: number, dtype: int64

          date  item  price  number
r1  2020/05/01   foo  12345     NaN
r2  2020/05/01   bar  23456     1.0
r3  2020/05/02  hoge   3456     2.0
r4  2020/05/03  piyo    456     3.0
r5  2020/05/04  fuga     56     NaN

          date  item  price  number
r2  2020/05/01   bar  23456       1
r3  2020/05/02  hoge   3456       2
r4  2020/05/03  piyo    456       3

          date  item  price        date  item  price  number  number
r2  2020/05/01   bar  23456  2020/05/01   bar  23456       1       1
r3  2020/05/02  hoge   3456  2020/05/02  hoge   3456       2       2
r4  2020/05/03  piyo    456  2020/05/03  piyo    456       3       3
```

<a id="markdown-要素に対する演算" name="要素に対する演算"></a>

### 要素に対する演算

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
print(df)

# 各要素に対する演算
print(df + 1)

# 各列に対する演算
print(
    df.apply(lambda c: c['r1']*c['r5'], axis=0)
)

# 各行に対する演算
print(
    df.apply(lambda r: pd.Series(dict(a=r['c1']+r['c10'], b=r['c1']*r['c10'])), axis=1)
)

```

```
# print(df)

column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49


# 各要素に対する演算

column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       1   2   3   4   5   6   7   8   9   10
r2      11  12  13  14  15  16  17  18  19   20
r3      21  22  23  24  25  26  27  28  29   30
r4      31  32  33  34  35  36  37  38  39   40
r5      41  42  43  44  45  46  47  48  49   50


# 各列に対する演算

column
c1       0
c2      41
c3      84
c4     129
c5     176
c6     225
c7     276
c8     329
c9     384
c10    441
dtype: int64


# 各行に対する演算

        a     b
index
r1      9     0
r2     29   190
r3     49   580
r4     69  1170
r5     89  1960
```

<a id="markdown-転置" name="転置"></a>

### 転置

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
print(df.shape) # タプル (行数, 列数)
print(df)

df = df.T

print(df.shape) # タプル (行数, 列数)
print(df)
```

```
(5, 10)

column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

# 転置
(10, 5)

index   r1  r2  r3  r4  r5
column
c1       0  10  20  30  40
c2       1  11  21  31  41
c3       2  12  22  32  42
c4       3  13  23  33  43
c5       4  14  24  34  44
c6       5  15  25  35  45
c7       6  16  26  36  46
c8       7  17  27  37  47
c9       8  18  28  38  48
c10      9  19  29  39  49
```

<a id="markdown-値の置換" name="値の置換"></a>

### 値の置換

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
print(df)

# 1要素だけ置換
df = df.replace(0, 999)
# print(df)

# 複数要素を置換
df = df.replace({10: 1000, 20: 2000})
# print(df)
df = df.replace([30, 40], [3000, 4000])
# print(df)
df = df.replace([1, 2, 3, 4], 0)
print(df)
```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

column    c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       999   0   0   0   0   5   6   7   8    9
r2      1000  11  12  13  14  15  16  17  18   19
r3      2000  21  22  23  24  25  26  27  28   29
r4      3000  31  32  33  34  35  36  37  38   39
r5      4000  41  42  43  44  45  46  47  48   49
```

<a id="markdown-欠損値" name="欠損値"></a>

#### 欠損値

<a id="markdown-欠損値に置き換える" name="欠損値に置き換える"></a>

##### 欠損値に置き換える

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
print(df)


df = df.replace([0, 1, 10], np.nan) # NaNを含む列はfloat型になる
print(df)
```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49

column    c1    c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       NaN   NaN   2   3   4   5   6   7   8    9
r2       NaN  11.0  12  13  14  15  16  17  18   19
r3      20.0  21.0  22  23  24  25  26  27  28   29
r4      30.0  31.0  32  33  34  35  36  37  38   39
r5      40.0  41.0  42  43  44  45  46  47  48   49
```

<a id="markdown-欠損値を置き換える" name="欠損値を置き換える"></a>

##### 欠損値を置き換える

<a id="markdown-定数で置換" name="定数で置換"></a>

###### 定数で置換

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df = df.replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.nan)
print(df)

df.fillna(-1)

df.fillna({'c1': -1, 'c2': -2, 'c3': -3})
```

```
# df
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
r2       NaN  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# df.fillna(-1)
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      -1.0  -1.0  -1.0  -1.0  -1.0  -1.0  -1.0  -1.0  -1.0  -1.0
r2      -1.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# df.fillna({'c1': -1, 'c2': -2, 'c3': -3})
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      -1.0  -2.0  -3.0   NaN   NaN   NaN   NaN   NaN   NaN   NaN
r2      -1.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0
```

<a id="markdown-統計量で置換" name="統計量で置換"></a>

###### 統計量で置換

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df = df.replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.nan)
print(df)

df.fillna(df.mean(numeric_only=True)) # 列ごとの平均値で置換
df.fillna(df.median(numeric_only=True)) # 列ごとの中央値（要素数が偶数の場合は中央2要素の平均値）で置換
df.fillna(df.mode(numeric_only=True).iloc[0]) # 列ごとの最頻値で置換
```

```
# df
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
r2       NaN  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# 列ごとの平均値で置換
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      30.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0
r2      30.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# 列ごとの中央値で置換
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      30.0  26.0  27.0  28.0  29.0  30.0  31.0  32.0  33.0  34.0
r2      30.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0

# 列ごとの最頻値で置換
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      20.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r2      20.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
r3      20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
r4      30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
r5      40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0
```

<a id="markdown-前後の要素で置換" name="前後の要素で置換"></a>

###### 前後の要素で置換

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))
df = df.replace([0, 11, 22, 33, 44, 15, 17, 18, 19, 25, 26, 27, 28, 29, 36, 37, 38, 39], np.nan)
print(df)

df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)

df.fillna(method='bfill')
df.fillna(method='bfill', limit=2)
```

```
# df
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
r2      10.0   NaN  12.0  13.0  14.0   NaN  16.0   NaN   NaN   NaN
r3      20.0  21.0   NaN  23.0  24.0   NaN   NaN   NaN   NaN   NaN
r4      30.0  31.0  32.0   NaN  34.0  35.0   NaN   NaN   NaN   NaN
r5      40.0  41.0  42.0  43.0   NaN  45.0  46.0  47.0  48.0  49.0

# df.fillna(method='ffill')
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
r2      10.0   1.0  12.0  13.0  14.0   5.0  16.0   7.0   8.0   9.0
r3      20.0  21.0  12.0  23.0  24.0   5.0  16.0   7.0   8.0   9.0
r4      30.0  31.0  32.0  23.0  34.0  35.0  16.0   7.0   8.0   9.0
r5      40.0  41.0  42.0  43.0  34.0  45.0  46.0  47.0  48.0  49.0

# df.fillna(method='ffill', limit=2)
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1       NaN   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
r2      10.0   1.0  12.0  13.0  14.0   5.0  16.0   7.0   8.0   9.0
r3      20.0  21.0  12.0  23.0  24.0   5.0  16.0   7.0   8.0   9.0
r4      30.0  31.0  32.0  23.0  34.0  35.0  16.0   NaN   NaN   NaN
r5      40.0  41.0  42.0  43.0  34.0  45.0  46.0  47.0  48.0  49.0

# df.fillna(method='bfill')
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      10.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
r2      10.0  21.0  12.0  13.0  14.0  35.0  16.0  47.0  48.0  49.0
r3      20.0  21.0  32.0  23.0  24.0  35.0  46.0  47.0  48.0  49.0
r4      30.0  31.0  32.0  43.0  34.0  35.0  46.0  47.0  48.0  49.0
r5      40.0  41.0  42.0  43.0   NaN  45.0  46.0  47.0  48.0  49.0

# df.fillna(method='bfill', limit=2)
column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      10.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
r2      10.0  21.0  12.0  13.0  14.0  35.0  16.0   NaN   NaN   NaN
r3      20.0  21.0  32.0  23.0  24.0  35.0  46.0  47.0  48.0  49.0
r4      30.0  31.0  32.0  43.0  34.0  35.0  46.0  47.0  48.0  49.0
r5      40.0  41.0  42.0  43.0   NaN  45.0  46.0  47.0  48.0  49.0
```

<a id="markdown-特徴量エンジニアリング" name="特徴量エンジニアリング"></a>

### 特徴量エンジニアリング

<a id="markdown-ビニング処理ビン分割" name="ビニング処理ビン分割"></a>

#### ビニング処理（ビン分割）

```py
import pandas as pd

df = pd.DataFrame({
    'item': ['foo1', 'bar1', 'hoge1', 'piyo1', 'fuga1', 'foo2', 'bar2', 'hoge2', 'piyo2', 'fuga2'],
    'price': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    })
print(df)

# 値で3分割
# df.priceはpandas.Seriesになる（df[['price']]だとNG）
df_cut1 = pd.cut(df.price, 3, labels=['small', 'medium', 'large'])
print(df_cut1)

# 境界値を指定して分割
df_cut2 = pd.cut(df.price, [2, 5, 50, 100], labels=['small', 'medium', 'large'])
print(df_cut2)

# 各ビンに含まれる要素数が等しくなるように分割
df_cut3 = pd.qcut(df.price, 3, labels=['small', 'medium', 'large'])
print(df_cut3)

# 4分位数で分割
df_qua, bins = pd.qcut(df.price, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], retbins=True)
print(df_qua)
print(bins)
```

```
# df

    item  price
0   foo1      2
1   bar1      3
2  hoge1      5
3  piyo1      8
4  fuga1     13
5   foo2     21
6   bar2     34
7  hoge2     55
8  piyo2     89
9  fuga2    144


# 値で3分割

0     small
1     small
2     small
3     small
4     small
5     small
6     small
7    medium
8    medium
9     large
Name: price, dtype: category
Categories (3, object): [small < medium < large]


# 境界値を指定して分割

0       NaN
1     small
2     small
3    medium
4    medium
5    medium
6    medium
7     large
8     large
9       NaN
Name: price, dtype: category
Categories (3, object): [small < medium < large]


# 各ビンに含まれる要素数が等しくなるように分割

0     small
1     small
2     small
3     small
4    medium
5    medium
6    medium
7     large
8     large
9     large
Name: price, dtype: category
Categories (3, object): [small < medium < large]


# 4分位数で分割

0    Q1
1    Q1
2    Q1
3    Q2
4    Q2
5    Q3
6    Q3
7    Q4
8    Q4
9    Q4
Name: price, dtype: category
Categories (4, object): [Q1 < Q2 < Q3 < Q4]

[  2.     5.75  17.    49.75 144.  ]
```

<a id="markdown-正規化" name="正規化"></a>

#### 正規化

<a id="markdown-最小値-0最大値-1" name="最小値-0最大値-1"></a>

##### 最小値 0、最大値 1

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

print(df)

# 各列ごと
print((df - df.min()) / (df.max() - df.min()))

# 全要素に対する
print((df - df.values.min()) / (df.values.max() - df.values.min()))
```

```
# df

column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49


# 各列ごと

column    c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
r1      0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
r2      0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25
r3      0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50
r4      0.75  0.75  0.75  0.75  0.75  0.75  0.75  0.75  0.75  0.75
r5      1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00


# 全要素

column        c1        c2        c3        c4        c5        c6        c7        c8        c9       c10
index
r1      0.000000  0.020408  0.040816  0.061224  0.081633  0.102041  0.122449  0.142857  0.163265  0.183673
r2      0.204082  0.224490  0.244898  0.265306  0.285714  0.306122  0.326531  0.346939  0.367347  0.387755
r3      0.408163  0.428571  0.448980  0.469388  0.489796  0.510204  0.530612  0.551020  0.571429  0.591837
r4      0.612245  0.632653  0.653061  0.673469  0.693878  0.714286  0.734694  0.755102  0.775510  0.795918
r5      0.816327  0.836735  0.857143  0.877551  0.897959  0.918367  0.938776  0.959184  0.979592  1.000000
```

<a id="markdown-平均-0分散-1" name="平均-0分散-1"></a>

##### 平均 0、分散 1

```py
import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(50).reshape(5, 10), index=pd.Index(['r{}'.format(x+1) for x in range(5)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(10)], name= 'column'))

print(df)

# 各列ごと
print((df - df.mean()) / df.std())

# 全要素
print((df - df.values.mean()) / df.values.std())
```

```
column  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
r1       0   1   2   3   4   5   6   7   8    9
r2      10  11  12  13  14  15  16  17  18   19
r3      20  21  22  23  24  25  26  27  28   29
r4      30  31  32  33  34  35  36  37  38   39
r5      40  41  42  43  44  45  46  47  48   49


# 各列ごと

column        c1        c2        c3        c4        c5        c6        c7        c8        c9       c10
index
r1     -1.264911 -1.264911 -1.264911 -1.264911 -1.264911 -1.264911 -1.264911 -1.264911 -1.264911 -1.264911
r2     -0.632456 -0.632456 -0.632456 -0.632456 -0.632456 -0.632456 -0.632456 -0.632456 -0.632456 -0.632456
r3      0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
r4      0.632456  0.632456  0.632456  0.632456  0.632456  0.632456  0.632456  0.632456  0.632456  0.632456
r5      1.264911  1.264911  1.264911  1.264911  1.264911  1.264911  1.264911  1.264911  1.264911  1.264911


# 全要素

column        c1        c2        c3        c4        c5        c6        c7        c8        c9       c10
index
r1     -1.697749 -1.628453 -1.559158 -1.489862 -1.420566 -1.351270 -1.281974 -1.212678 -1.143382 -1.074086
r2     -1.004790 -0.935495 -0.866199 -0.796903 -0.727607 -0.658311 -0.589015 -0.519719 -0.450423 -0.381127
r3     -0.311832 -0.242536 -0.173240 -0.103944 -0.034648  0.034648  0.103944  0.173240  0.242536  0.311832
r4      0.381127  0.450423  0.519719  0.589015  0.658311  0.727607  0.796903  0.866199  0.935495  1.004790
r5      1.074086  1.143382  1.212678  1.281974  1.351270  1.420566  1.489862  1.559158  1.628453  1.697749
```

<a id="markdown-カテゴリ変数のエンコーディング" name="カテゴリ変数のエンコーディング"></a>

#### カテゴリ変数のエンコーディング

```py
import pandas as pd

df = pd.DataFrame({
    'date': ['2020/05/01', '2020/05/01', '2020/05/01', '2020/05/01', '2020/05/02', '2020/05/02', '2020/05/02', '2020/05/03', '2020/05/03', '2020/05/04'],
    'price': [12345, 23456, 3456, 456, 56, 56, 7, 8, 9, 0]
    })
print(df)

# カテゴリ変数の値の種類を確認する
counts = df['date'].value_counts()
print(counts)

# ワンホットエンコーディング（ダミー変数）
df_dummy = pd.get_dummies(df['date'])
print(df_dummy)

df_dummy = pd.get_dummies(df, drop_first=True, columns=['date', 'price'])
print(df_dummy)

# ラベルエンコーディング
df_label = df.copy(deep=True)
df_label['date_cat'] = df_label['date'].astype('category')
df_label.dtypes
df_label['date_label'] = df_label['date_cat'].cat.codes
print(df_label)

# カウントエンコーディング
df_count = df.copy(deep=True)
df_count['date_count'] = df_count['date'].map(df_count.groupby('date').price.count())
print(df_count)

# ラベルカウントエンコーディング
df_labelcount = df.copy(deep=True)
df_labelcount['count_rank'] = df_labelcount['date'].map(
    df_labelcount.groupby('date')['price'].count().rank(ascending=False)
).astype('int')
print(df_labelcount)

# ターゲットエンコーディング
df_target = df.copy(deep=True)
df_target['target_enc'] = df_target['date'].map(df_target.groupby('date').price.mean())
print(df_target)


```

```
         date  price
0  2020/05/01  12345
1  2020/05/01  23456
2  2020/05/01   3456
3  2020/05/01    456
4  2020/05/02     56
5  2020/05/02     56
6  2020/05/02      7
7  2020/05/03      8
8  2020/05/03      9
9  2020/05/04      0


# カテゴリカルデータの値の種類を確認する

2020/05/01    4
2020/05/02    3
2020/05/03    2
2020/05/04    1
Name: date, dtype: int64


# ワンホットエンコーディング（ダミー変数）

   2020/05/01  2020/05/02  2020/05/03  2020/05/04
0           1           0           0           0
1           1           0           0           0
2           1           0           0           0
3           1           0           0           0
4           0           1           0           0
5           0           1           0           0
6           0           1           0           0
7           0           0           1           0
8           0           0           1           0
9           0           0           0           1

   date_2020/05/02  date_2020/05/03  date_2020/05/04  price_7  price_8  ...  price_56  price_456  price_3456  price_12345  price_23456
0                0                0                0        0        0  ...         0          0           0            1            0
1                0                0                0        0        0  ...         0          0           0            0            1
2                0                0                0        0        0  ...         0          0           1            0            0
3                0                0                0        0        0  ...         0          1           0            0            0
4                1                0                0        0        0  ...         1          0           0            0            0
5                1                0                0        0        0  ...         1          0           0            0            0
6                1                0                0        1        0  ...         0          0           0            0            0
7                0                1                0        0        1  ...         0          0           0            0            0
8                0                1                0        0        0  ...         0          0           0            0            0
9                0                0                1        0        0  ...         0          0           0            0            0

[10 rows x 11 columns]


# ラベルエンコーディング

date          object
price          int64
date_cat    category
dtype: object

         date  price    date_cat  date_label
0  2020/05/01  12345  2020/05/01           0
1  2020/05/01  23456  2020/05/01           0
2  2020/05/01   3456  2020/05/01           0
3  2020/05/01    456  2020/05/01           0
4  2020/05/02     56  2020/05/02           1
5  2020/05/02     56  2020/05/02           1
6  2020/05/02      7  2020/05/02           1
7  2020/05/03      8  2020/05/03           2
8  2020/05/03      9  2020/05/03           2
9  2020/05/04      0  2020/05/04           3


# カウントエンコーディング

         date  price  date_count
0  2020/05/01  12345           4
1  2020/05/01  23456           4
2  2020/05/01   3456           4
3  2020/05/01    456           4
4  2020/05/02     56           3
5  2020/05/02     56           3
6  2020/05/02      7           3
7  2020/05/03      8           2
8  2020/05/03      9           2
9  2020/05/04      0           1


# ラベルカウントエンコーディング

         date  price  count_rank
0  2020/05/01  12345           1
1  2020/05/01  23456           1
2  2020/05/01   3456           1
3  2020/05/01    456           1
4  2020/05/02     56           2
5  2020/05/02     56           2
6  2020/05/02      7           2
7  2020/05/03      8           3
8  2020/05/03      9           3
9  2020/05/04      0           4


# ターゲットエンコーディング

0  2020/05/01  12345  9928.250000
1  2020/05/01  23456  9928.250000
2  2020/05/01   3456  9928.250000
3  2020/05/01    456  9928.250000
4  2020/05/02     56    39.666667
5  2020/05/02     56    39.666667
6  2020/05/02      7    39.666667
7  2020/05/03      8     8.500000
8  2020/05/03      9     8.500000
9  2020/05/04      0     0.000000


```

<a id="markdown-データ結合" name="データ結合"></a>

## データ結合

<a id="markdown-inner-join" name="inner-join"></a>

### inner join

```py
import pandas as pd

df1 = pd.DataFrame({
    'key': ['a', 'b', 'c', 'd', 'e', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key': ['c', 'a', 'b'],
    'df2': range(3)
})

df1
df2

joined = pd.merge(df1, df2)
# joined = pd.merge(df1, df2, how='inner')
print(joined)

# indexを使って結合
joined_by_indexes = pd.merge(df1, df2, left_index=True, right_index=True)
# joined_by_index_and_data = pd.merge(df1, df2, left_index=True, right_on='key') # 片方だけインデックス、片方はカラム、という指定方法も可能
```

```
  key  df1
0   a    0
1   b    1
2   c    2
3   d    3
4   e    4
5   f    5

  key  df2
0   c    0
1   a    1
2   b    2

  key  df1  df2
0   a    0    1
1   b    1    2
2   c    2    0

# indexを使って結合
  key_x  df1 key_y  df2
0     a    0     c    0
1     b    1     a    1
2     c    2     b    2
```

```py
import pandas as pd

df1 = pd.DataFrame({
    'key1': ['a', 'b', 'c', 'd', 'e', 'f'],
    'key2': ['a', 'c', 'e', 'b', 'd', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key1': ['c', 'a', 'b'],
    'key2': ['c', 'b', 'a'],
    'df2': range(3)
})

df1
df2

joined1 = pd.merge(df1, df2, on='key2')
print(joined1)

joined2 = pd.merge(df1, df2, left_on='key1', right_on='key2')
print(joined2)

# 接尾辞（サフィックス）を、「_x」「_y」から変更する
joined3 = pd.merge(df1, df2, on='key1', suffixes=('_LEFT', '_RIGHT'))
print(joined3)
```

```
  key1 key2  df1
0    a    a    0
1    b    c    1
2    c    e    2
3    d    b    3
4    e    d    4
5    f    f    5

  key1 key2  df2
0    c    c    0
1    a    b    1
2    b    a    2

  key1_x key2  df1 key1_y  df2
0      a    a    0      b    2
1      b    c    1      c    0
2      d    b    3      a    1

  key1_x key2_x  df1 key1_y key2_y  df2
0      a      a    0      b      a    2
1      b      c    1      a      b    1
2      c      e    2      c      c    0

# 接尾辞（サフィックス）を、「_x」「_y」から変更する
  key1 key2_LEFT  df1 key2_RIGHT  df2
0    a         a    0          b    1
1    b         c    1          a    2
2    c         e    2          c    0
```

<a id="markdown-複数キー" name="複数キー"></a>

#### 複数キー

```py
import pandas as pd

df1 = pd.DataFrame({
    'key1': ['a', 'b', 'c', 'd', 'e', 'f'],
    'key2': ['c', 'a', 'c', 'b', 'd', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key1': ['c', 'a', 'b'],
    'key2': ['c', 'b', 'a'],
    'df2': range(3)
})

df1
df2

joined = pd.merge(df1, df2, on=['key1', 'key2'])
print(joined)
```

```
  key1 key2  df1  df2
0    b    a    1    2
1    c    c    2    0
```

<a id="markdown-left-join" name="left-join"></a>

### left join

```py
import pandas as pd

df1 = pd.DataFrame({
    'key': ['a', 'b', 'c', 'd', 'e', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key': ['c', 'a', 'b'],
    'df2': range(3)
})

df1
df2

joined = pd.merge(df1, df2, how='left')
print(joined)
```

```
  key  df1
0   a    0
1   b    1
2   c    2
3   d    3
4   e    4
5   f    5

  key  df2
0   c    0
1   a    1
2   b    2

  key  df1  df2
0   a    0  1.0
1   b    1  2.0
2   c    2  0.0
3   d    3  NaN
4   e    4  NaN
5   f    5  NaN
```

```py
import pandas as pd

df1 = pd.DataFrame({
    'key1': ['a', 'b', 'c', 'd', 'e', 'f'],
    'key2': ['a', 'c', 'e', 'b', 'd', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key1': ['c', 'a', 'b'],
    'key2': ['c', 'b', 'a'],
    'df2': range(3)
})

df1
df2

joined1 = pd.merge(df1, df2, how='left', on='key2')
print(joined1)

joined2 = pd.merge(df1, df2, how='left', left_on='key1', right_on='key2')
print(joined2)
```

```
  key1 key2  df1
0    a    a    0
1    b    c    1
2    c    e    2
3    d    b    3
4    e    d    4
5    f    f    5

  key1 key2  df2
0    c    c    0
1    a    b    1
2    b    a    2

  key1_x key2  df1 key1_y  df2
0      a    a    0      b  2.0
1      b    c    1      c  0.0
2      c    e    2    NaN  NaN
3      d    b    3      a  1.0
4      e    d    4    NaN  NaN
5      f    f    5    NaN  NaN

  key1_x key2_x  df1 key1_y key2_y  df2
0      a      a    0      b      a  2.0
1      b      c    1      a      b  1.0
2      c      e    2      c      c  0.0
3      d      b    3    NaN    NaN  NaN
4      e      d    4    NaN    NaN  NaN
5      f      f    5    NaN    NaN  NaN
```

<a id="markdown-right-join" name="right-join"></a>

### right join

```py
import pandas as pd

df1 = pd.DataFrame({
    'key': ['a', 'b', 'c', 'd', 'e', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key': ['c', 'a', 'b'],
    'df2': range(3)
})

df1
df2

joined = pd.merge(df1, df2, how='right')
print(joined)
```

```
  key  df1
0   a    0
1   b    1
2   c    2
3   d    3
4   e    4
5   f    5

  key  df2
0   c    0
1   a    1
2   b    2

  key  df1  df2
0   a    0    1
1   b    1    2
2   c    2    0
```

<a id="markdown-outer-join" name="outer-join"></a>

### outer join

```py
import pandas as pd

df1 = pd.DataFrame({
    'key': ['a', 'b', 'c', 'd', 'e', 'f'],
    'df1': range(6)
})
df2 = pd.DataFrame({
    'key': ['c', 'a', 'b'],
    'df2': range(3)
})

df1
df2

joined = pd.merge(df1, df2, how='outer')
print(joined)
```

```
  key  df1
0   a    0
1   b    1
2   c    2
3   d    3
4   e    4
5   f    5

  key  df2
0   c    0
1   a    1
2   b    2

  key  df1  df2
0   a    0  1.0
1   b    1  2.0
2   c    2  0.0
3   d    3  NaN
4   e    4  NaN
5   f    5  NaN
```

<a id="markdown-グループ化" name="グループ化"></a>

## グループ化

```py
import pandas as pd


data_url =  "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"


# df
df = pd.read_csv(data_url)
print(df)

# groupby
df_groupby = df.groupby('variety').mean()
df_groupby.sort_values('sepal.length', ascending=False)

# size
#   各グループの要素数
df.groupby(['variety']).size()

# agg
#   variety と petal.width をグルーピングして sepal.length に集約関数を適用
aggregation = {'sepal.length':['median',  'mean', 'min', 'max']}
df_agg = df.groupby(['variety', 'petal.width']).agg(aggregation).reset_index()
```

```
# df

     sepal.length  sepal.width  petal.length  petal.width    variety
0             5.1          3.5           1.4          0.2     Setosa
1             4.9          3.0           1.4          0.2     Setosa
2             4.7          3.2           1.3          0.2     Setosa
3             4.6          3.1           1.5          0.2     Setosa
4             5.0          3.6           1.4          0.2     Setosa
..            ...          ...           ...          ...        ...
145           6.7          3.0           5.2          2.3  Virginica
146           6.3          2.5           5.0          1.9  Virginica
147           6.5          3.0           5.2          2.0  Virginica
148           6.2          3.4           5.4          2.3  Virginica
149           5.9          3.0           5.1          1.8  Virginica

[150 rows x 5 columns]


# groupby

            sepal.length  sepal.width  petal.length  petal.width
variety
Virginica          6.588        2.974         5.552        2.026
Versicolor         5.936        2.770         4.260        1.326
Setosa             5.006        3.428         1.462        0.246


# size

variety
Setosa        50
Versicolor    50
Virginica     50
dtype: int64


# agg

       variety petal.width sepal.length
                                 median      mean  min  max
0       Setosa         0.1         4.90  4.820000  4.3  5.2
1       Setosa         0.2         5.00  4.972414  4.4  5.8
2       Setosa         0.3         5.00  4.971429  4.5  5.7
3       Setosa         0.4         5.40  5.300000  5.0  5.7
4       Setosa         0.5         5.10  5.100000  5.1  5.1
5       Setosa         0.6         5.00  5.000000  5.0  5.0
6   Versicolor         1.0         5.50  5.414286  4.9  6.0
7   Versicolor         1.1         5.50  5.400000  5.1  5.6
8   Versicolor         1.2         5.80  5.780000  5.5  6.1
9   Versicolor         1.3         5.70  5.884615  5.5  6.6
10  Versicolor         1.4         6.60  6.357143  5.2  7.0
11  Versicolor         1.5         6.25  6.190000  5.4  6.9
12  Versicolor         1.6         6.00  6.100000  6.0  6.3
13  Versicolor         1.7         6.70  6.700000  6.7  6.7
14  Versicolor         1.8         5.90  5.900000  5.9  5.9
15   Virginica         1.4         6.10  6.100000  6.1  6.1
16   Virginica         1.5         6.15  6.150000  6.0  6.3
17   Virginica         1.6         7.20  7.200000  7.2  7.2
18   Virginica         1.7         4.90  4.900000  4.9  4.9
19   Virginica         1.8         6.30  6.445455  5.9  7.3
20   Virginica         1.9         6.30  6.340000  5.8  7.4
21   Virginica         2.0         6.50  6.650000  5.6  7.9
22   Virginica         2.1         6.85  6.916667  6.4  7.6
23   Virginica         2.2         6.50  6.866667  6.4  7.7
24   Virginica         2.3         6.85  6.912500  6.2  7.7
25   Virginica         2.4         6.30  6.266667  5.8  6.7
26   Virginica         2.5         6.70  6.733333  6.3  7.2
```

<a id="markdown-時系列データ-1" name="時系列データ-1"></a>

## 時系列データ

```py
import numpy as np
import pandas as pd

import datetime

start_date = datetime.datetime(2020, 6, 1)
days = 365
ncol = 10

arr = np.arange(days * ncol).reshape(days, ncol)
idx = pd.Index([start_date + datetime.timedelta(days=x) for x in range(days)], name = 'index')
col = pd.Index(['c{}'.format(x+1) for x in range(ncol)], name= 'column')
df = pd.DataFrame(arr, index=idx, columns=col)
print(df.head())
print(df.tail())
```

```
c4  c5  c6  c7  c8  c9  c10
index
2020-06-01   0   1   2   3   4   5   6   7   8    9
2020-06-02  10  11  12  13  14  15  16  17  18   19
2020-06-03  20  21  22  23  24  25  26  27  28   29
2020-06-04  30  31  32  33  34  35  36  37  38   39
2020-06-05  40  41  42  43  44  45  46  47  48   49

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2021-05-27  3600  3601  3602  3603  3604  3605  3606  3607  3608  3609
2021-05-28  3610  3611  3612  3613  3614  3615  3616  3617  3618  3619
2021-05-29  3620  3621  3622  3623  3624  3625  3626  3627  3628  3629
2021-05-30  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-31  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
```

<a id="markdown-頻度の指定オフセットエイリアス" name="頻度の指定オフセットエイリアス"></a>

### 頻度の指定（オフセットエイリアス）

| エイリアス  | 説明                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------ |
| B           | 営業日（月曜 - 金曜）                                                                      |
| C           | 独自の営業日                                                                               |
| D           | 日                                                                                         |
| W （W-SUN） | 週（月曜-日曜）                                                                            |
|             | W-MON（火曜-月曜）, W-TUE, W-WED, W-THU, W-FRI, W-SAT                                      |
| M           | 月末                                                                                       |
| SM          | 15 日と月末                                                                                |
| BM          | 月の最終営業日                                                                             |
| CBM         | 独自の営業月末                                                                             |
| MS          | 月初                                                                                       |
| SMS         | 月初と 15 日                                                                               |
| BMS         | 月の第 1 営業日                                                                            |
| CBMS        | 独自の営業月の第 1 営業日                                                                  |
| Q           | 四半期末                                                                                   |
|             | Q-JAN, Q-APR, Q-JUL, Q-OCT（2 月～ 4 月，5 月～ 7 月 ，8 月～ 10 月，11 月～ 1 月）        |
|             | Q-FEB, Q-MAY, Q-AUG, Q-NOV / Q-MAR, Q-JUN, Q-SEP, Q-DEC                                    |
| BQ          | 四半期の最終営業日                                                                         |
| QS          | 四半期初                                                                                   |
| BQS         | 四半期の第 1 営業日                                                                        |
| A, Y        | 年末                                                                                       |
|             | A-JAN（2 月～ 1 月）, A-FEB, A-MAR, A-APR, A-MAY, A-JUN, A-JUL, A-AUG, A-SEP, A-OCT, A-NOV |
| BA, BY      | 年の最終営業日                                                                             |
| AS, YS      | 年初                                                                                       |
| BAS, BYS    | 年の第 1 営業日                                                                            |
| WOM-1MON    | 月の第 1 月曜日 （第 4 金曜日なら`WOM-4FRI`）                                              |
| BH          | 業務時間                                                                                   |
| H           | 時                                                                                         |
| T, min      | 分                                                                                         |
| S           | 秒                                                                                         |
| L, ms       | ミリ秒                                                                                     |
| U, us       | マイクロ秒                                                                                 |
| N           | ナノ秒                                                                                     |

```py
import datetime
import numpy as np
import pandas as pd

start_date = datetime.datetime(2020, 6, 1)
days = 365
ncol = 10
df = pd.DataFrame(np.arange(days * ncol).reshape(days, ncol), index=pd.Index([start_date + datetime.timedelta(days=x) for x in range(days)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(ncol)], name= 'column'))


# 3月終わりの年度
df.resample(rule='A-MAR').sum().head()

# 第1月曜日
df.resample(rule='WOM-1MON').sum().head()

# 複数エイリアスの組み合わせ
df.resample(rule='1D2H3T4S5L6U7N').min()

# 独自の営業日を定義
weekmask = 'Tue Wed Thu Fri Sat' # 日曜・月曜を休業日とする
holidays = ['2020-06-01', datetime.datetime(2020, 6, 2), np.datetime64('2020-06-03'), '2020-06-04', '2020-06-05', '2020-06-06'] # 6/1～6を休業日とする
bday = pd.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask)
dt = datetime.datetime(2020, 5, 31)
print(dt + bday) # 2020-06-09 00:00:00
```

```
# 3月終わりの年度

column          c1      c2      c3      c4      c5      c6      c7      c8      c9     c10
index
2021-03-31  460560  460864  461168  461472  461776  462080  462384  462688  462992  463296
2022-03-31  203740  203801  203862  203923  203984  204045  204106  204167  204228  204289


# 第1月曜日

column         c1     c2     c3     c4     c5     c6     c7     c8     c9    c10
index
2020-06-01   5950   5985   6020   6055   6090   6125   6160   6195   6230   6265
2020-07-06  13580  13608  13636  13664  13692  13720  13748  13776  13804  13832
2020-08-03  28000  28035  28070  28105  28140  28175  28210  28245  28280  28315
2020-09-07  31220  31248  31276  31304  31332  31360  31388  31416  31444  31472
2020-10-05  39060  39088  39116  39144  39172  39200  39228  39256  39284  39312


# 複数エイリアスの組み合わせ

column                           c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00.000000000     0     1     2     3     4     5     6     7     8     9
2020-06-02 02:03:04.005006007    20    21    22    23    24    25    26    27    28    29
2020-06-03 04:06:08.010012014    30    31    32    33    34    35    36    37    38    39
2020-06-04 06:09:12.015018021    40    41    42    43    44    45    46    47    48    49
2020-06-05 08:12:16.020024028    50    51    52    53    54    55    56    57    58    59
...                             ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-26 06:55:05.656988317  3600  3601  3602  3603  3604  3605  3606  3607  3608  3609
2021-05-27 08:58:09.661994324  3610  3611  3612  3613  3614  3615  3616  3617  3618  3619
2021-05-28 11:01:13.667000331  3620  3621  3622  3623  3624  3625  3626  3627  3628  3629
2021-05-29 13:04:17.672006338  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 15:07:21.677012345  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649

[336 rows x 10 columns]
```

<a id="markdown-ラグ" name="ラグ"></a>

### ラグ

```py
import datetime
import numpy as np
import pandas as pd

start_date = datetime.datetime(2020, 6, 1)
days = 365
ncol = 10
df = pd.DataFrame(np.arange(days * ncol).reshape(days, ncol), index=pd.Index([start_date + datetime.timedelta(days=x) for x in range(days)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(ncol)], name= 'column'))

# 下方向に1行
print(df.shift(1).head())
print(df.shift(1).tail())

# 上方向に7行
print(df.shift(-7).head())
print(df.shift(-7).tail())

# 3日
print(df.shift(freq='3D'))
```

```
# 下方向に1行
column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
2020-06-02   0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
2020-06-03  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
2020-06-04  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
2020-06-05  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0

column          c1      c2      c3      c4      c5      c6      c7      c8      c9     c10
index
2021-05-27  3590.0  3591.0  3592.0  3593.0  3594.0  3595.0  3596.0  3597.0  3598.0  3599.0
2021-05-28  3600.0  3601.0  3602.0  3603.0  3604.0  3605.0  3606.0  3607.0  3608.0  3609.0
2021-05-29  3610.0  3611.0  3612.0  3613.0  3614.0  3615.0  3616.0  3617.0  3618.0  3619.0
2021-05-30  3620.0  3621.0  3622.0  3623.0  3624.0  3625.0  3626.0  3627.0  3628.0  3629.0
2021-05-31  3630.0  3631.0  3632.0  3633.0  3634.0  3635.0  3636.0  3637.0  3638.0  3639.0


# 上方向に7行

column         c1     c2     c3     c4     c5     c6     c7     c8     c9    c10
index
2020-06-01   70.0   71.0   72.0   73.0   74.0   75.0   76.0   77.0   78.0   79.0
2020-06-02   80.0   81.0   82.0   83.0   84.0   85.0   86.0   87.0   88.0   89.0
2020-06-03   90.0   91.0   92.0   93.0   94.0   95.0   96.0   97.0   98.0   99.0
2020-06-04  100.0  101.0  102.0  103.0  104.0  105.0  106.0  107.0  108.0  109.0
2020-06-05  110.0  111.0  112.0  113.0  114.0  115.0  116.0  117.0  118.0  119.0

column      c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
2021-05-27 NaN NaN NaN NaN NaN NaN NaN NaN NaN  NaN
2021-05-28 NaN NaN NaN NaN NaN NaN NaN NaN NaN  NaN
2021-05-29 NaN NaN NaN NaN NaN NaN NaN NaN NaN  NaN
2021-05-30 NaN NaN NaN NaN NaN NaN NaN NaN NaN  NaN
2021-05-31 NaN NaN NaN NaN NaN NaN NaN NaN NaN  NaN


# 3日

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-04     0     1     2     3     4     5     6     7     8     9
2020-06-05    10    11    12    13    14    15    16    17    18    19
2020-06-06    20    21    22    23    24    25    26    27    28    29
2020-06-07    30    31    32    33    34    35    36    37    38    39
2020-06-08    40    41    42    43    44    45    46    47    48    49
...          ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30  3600  3601  3602  3603  3604  3605  3606  3607  3608  3609
2021-05-31  3610  3611  3612  3613  3614  3615  3616  3617  3618  3619
2021-06-01  3620  3621  3622  3623  3624  3625  3626  3627  3628  3629
2021-06-02  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-06-03  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649

[365 rows x 10 columns]
```

<a id="markdown-時系列データのグループ化" name="時系列データのグループ化"></a>

### 時系列データのグループ化

```py
import datetime
import numpy as np
import pandas as pd

start_date = datetime.datetime(2020, 6, 1)
days = 365
ncol = 10
df = pd.DataFrame(np.arange(days * ncol).reshape(days, ncol), index=pd.Index([start_date + datetime.timedelta(days=x) for x in range(days)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(ncol)], name= 'column'))

# A: 年
df.resample(rule='A').min()

# Q: 四半期
df.resample(rule='Q').median()

# M: 月
round(df.resample(rule='M').mean())

# W: 週
df.resample(rule='W').sum().head()

# 先頭の値
df.resample('Q', label='left', closed='left').first()
df.resample('Q', label='right', closed='right').first()

# 末尾の値
df.resample('M').last()

# 個数
df.resample('M').count()

# OHLC（Open: 始値、High: 高値、Low: 安値、Close: 終値）
df.resample('M').ohlc()
```

```
# A: 年

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-12-31     0     1     2     3     4     5     6     7     8     9
2021-12-31  2140  2141  2142  2143  2144  2145  2146  2147  2148  2149


# Q: 四半期

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-30   145   146   147   148   149   150   151   152   153   154
2020-09-30   755   756   757   758   759   760   761   762   763   764
2020-12-31  1675  1676  1677  1678  1679  1680  1681  1682  1683  1684
2021-03-31  2585  2586  2587  2588  2589  2590  2591  2592  2593  2594
2021-06-30  3340  3341  3342  3343  3344  3345  3346  3347  3348  3349


# M: 月

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-30   145   146   147   148   149   150   151   152   153   154
2020-07-31   450   451   452   453   454   455   456   457   458   459
2020-08-31   760   761   762   763   764   765   766   767   768   769
2020-09-30  1065  1066  1067  1068  1069  1070  1071  1072  1073  1074
2020-10-31  1370  1371  1372  1373  1374  1375  1376  1377  1378  1379
2020-11-30  1675  1676  1677  1678  1679  1680  1681  1682  1683  1684
2020-12-31  1980  1981  1982  1983  1984  1985  1986  1987  1988  1989
2021-01-31  2290  2291  2292  2293  2294  2295  2296  2297  2298  2299
2021-02-28  2585  2586  2587  2588  2589  2590  2591  2592  2593  2594
2021-03-31  2880  2881  2882  2883  2884  2885  2886  2887  2888  2889
2021-04-30  3185  3186  3187  3188  3189  3190  3191  3192  3193  3194
2021-05-31  3490  3491  3492  3493  3494  3495  3496  3497  3498  3499


# W: 週

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-07   210   217   224   231   238   245   252   259   266   273
2020-06-14   700   707   714   721   728   735   742   749   756   763
2020-06-21  1190  1197  1204  1211  1218  1225  1232  1239  1246  1253
2020-06-28  1680  1687  1694  1701  1708  1715  1722  1729  1736  1743
2020-07-05  2170  2177  2184  2191  2198  2205  2212  2219  2226  2233


# 先頭の値

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-30     0     1     2     3     4     5     6     7     8     9
2020-09-30   300   301   302   303   304   305   306   307   308   309
2020-12-31  1220  1221  1222  1223  1224  1225  1226  1227  1228  1229
2021-03-31  2140  2141  2142  2143  2144  2145  2146  2147  2148  2149
2021-06-30  3040  3041  3042  3043  3044  3045  3046  3047  3048  3049

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-03-31     0     1     2     3     4     5     6     7     8     9
2020-06-30   290   291   292   293   294   295   296   297   298   299
2020-09-30  1210  1211  1212  1213  1214  1215  1216  1217  1218  1219
2020-12-31  2130  2131  2132  2133  2134  2135  2136  2137  2138  2139
2021-03-31  3030  3031  3032  3033  3034  3035  3036  3037  3038  3039

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-30     0     1     2     3     4     5     6     7     8     9
2020-09-30   300   301   302   303   304   305   306   307   308   309
2020-12-31  1220  1221  1222  1223  1224  1225  1226  1227  1228  1229
2021-03-31  2140  2141  2142  2143  2144  2145  2146  2147  2148  2149
2021-06-30  3040  3041  3042  3043  3044  3045  3046  3047  3048  3049



# 末尾の値

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-30   290   291   292   293   294   295   296   297   298   299
2020-07-31   600   601   602   603   604   605   606   607   608   609
2020-08-31   910   911   912   913   914   915   916   917   918   919
2020-09-30  1210  1211  1212  1213  1214  1215  1216  1217  1218  1219
2020-10-31  1520  1521  1522  1523  1524  1525  1526  1527  1528  1529
2020-11-30  1820  1821  1822  1823  1824  1825  1826  1827  1828  1829
2020-12-31  2130  2131  2132  2133  2134  2135  2136  2137  2138  2139
2021-01-31  2440  2441  2442  2443  2444  2445  2446  2447  2448  2449
2021-02-28  2720  2721  2722  2723  2724  2725  2726  2727  2728  2729
2021-03-31  3030  3031  3032  3033  3034  3035  3036  3037  3038  3039
2021-04-30  3330  3331  3332  3333  3334  3335  3336  3337  3338  3339
2021-05-31  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649


# 個数

column      c1  c2  c3  c4  c5  c6  c7  c8  c9  c10
index
2020-06-30  30  30  30  30  30  30  30  30  30   30
2020-07-31  31  31  31  31  31  31  31  31  31   31
2020-08-31  31  31  31  31  31  31  31  31  31   31
2020-09-30  30  30  30  30  30  30  30  30  30   30
2020-10-31  31  31  31  31  31  31  31  31  31   31
2020-11-30  30  30  30  30  30  30  30  30  30   30
2020-12-31  31  31  31  31  31  31  31  31  31   31
2021-01-31  31  31  31  31  31  31  31  31  31   31
2021-02-28  28  28  28  28  28  28  28  28  28   28
2021-03-31  31  31  31  31  31  31  31  31  31   31
2021-04-30  30  30  30  30  30  30  30  30  30   30
2021-05-31  31  31  31  31  31  31  31  31  31   31


# OHLC（Open: 始値、High: 高値、Low: 安値、Close: 終値）

column        c1                      c2                      c3  ...    c8    c9                     c10
            open  high   low close  open  high   low close  open  ... close  open  high   low close  open  high   low close
index                                                             ...
2020-06-30     0   290     0   290     1   291     1   291     2  ...   297     8   298     8   298     9   299     9   299
2020-07-31   300   600   300   600   301   601   301   601   302  ...   607   308   608   308   608   309   609   309   609
2020-08-31   610   910   610   910   611   911   611   911   612  ...   917   618   918   618   918   619   919   619   919
2020-09-30   920  1210   920  1210   921  1211   921  1211   922  ...  1217   928  1218   928  1218   929  1219   929  1219
2020-10-31  1220  1520  1220  1520  1221  1521  1221  1521  1222  ...  1527  1228  1528  1228  1528  1229  1529  1229  1529
2020-11-30  1530  1820  1530  1820  1531  1821  1531  1821  1532  ...  1827  1538  1828  1538  1828  1539  1829  1539  1829
2020-12-31  1830  2130  1830  2130  1831  2131  1831  2131  1832  ...  2137  1838  2138  1838  2138  1839  2139  1839  2139
2021-01-31  2140  2440  2140  2440  2141  2441  2141  2441  2142  ...  2447  2148  2448  2148  2448  2149  2449  2149  2449
2021-02-28  2450  2720  2450  2720  2451  2721  2451  2721  2452  ...  2727  2458  2728  2458  2728  2459  2729  2459  2729
2021-03-31  2730  3030  2730  3030  2731  3031  2731  3031  2732  ...  3037  2738  3038  2738  3038  2739  3039  2739  3039
2021-04-30  3040  3330  3040  3330  3041  3331  3041  3331  3042  ...  3337  3048  3338  3048  3338  3049  3339  3049  3339
2021-05-31  3340  3640  3340  3640  3341  3641  3341  3641  3342  ...  3647  3348  3648  3348  3648  3349  3649  3349  3649

[12 rows x 40 columns]
```

<a id="markdown-時系列データの抽出" name="時系列データの抽出"></a>

### 時系列データの抽出

```py
import datetime
import numpy as np
import pandas as pd

start_date = datetime.datetime(2020, 6, 1)
days = 365
ncol = 10
df = pd.DataFrame(np.arange(days * ncol).reshape(days, ncol), index=pd.Index([start_date + datetime.timedelta(days=x) for x in range(days)], name = 'index'), columns=pd.Index(['c{}'.format(x+1) for x in range(ncol)], name= 'column'))

# 3日ごと
df.asfreq('3D')

# 毎週日曜日
df.asfreq('W')

# 毎週金曜日
df.asfreq('W-FRI').head()

# 毎時（欠損値は0埋め）
df.asfreq('H', fill_value=0)

# 毎時（欠損値は直前の行の値で補間）
df.asfreq('H', method='pad') # method='ffill'も同様
df.resample('H').ffill()

# 毎時（欠損値は直後の行の値で補間）
df.asfreq('H', method='backfill') # method='bfill'も同様
df.resample('H').bfill()

# 毎分（欠損値は前後の値から日付インデックスを基に線形補間）
df.asfreq('T').interpolate('time').head(60 * 24 + 1)
```

```
# 3日ごと

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01     0     1     2     3     4     5     6     7     8     9
2020-06-04    30    31    32    33    34    35    36    37    38    39
2020-06-07    60    61    62    63    64    65    66    67    68    69
2020-06-10    90    91    92    93    94    95    96    97    98    99
2020-06-13   120   121   122   123   124   125   126   127   128   129
...          ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-18  3510  3511  3512  3513  3514  3515  3516  3517  3518  3519
2021-05-21  3540  3541  3542  3543  3544  3545  3546  3547  3548  3549
2021-05-24  3570  3571  3572  3573  3574  3575  3576  3577  3578  3579
2021-05-27  3600  3601  3602  3603  3604  3605  3606  3607  3608  3609
2021-05-30  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639

[122 rows x 10 columns]


# 毎週日曜日

column        c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-07    60    61    62    63    64    65    66    67    68    69
2020-06-14   130   131   132   133   134   135   136   137   138   139
2020-06-21   200   201   202   203   204   205   206   207   208   209
2020-06-28   270   271   272   273   274   275   276   277   278   279
2020-07-05   340   341   342   343   344   345   346   347   348   349
2020-07-12   410   411   412   413   414   415   416   417   418   419
2020-07-19   480   481   482   483   484   485   486   487   488   489
2020-07-26   550   551   552   553   554   555   556   557   558   559
2020-08-02   620   621   622   623   624   625   626   627   628   629
2020-08-09   690   691   692   693   694   695   696   697   698   699
2020-08-16   760   761   762   763   764   765   766   767   768   769
2020-08-23   830   831   832   833   834   835   836   837   838   839
2020-08-30   900   901   902   903   904   905   906   907   908   909
2020-09-06   970   971   972   973   974   975   976   977   978   979
2020-09-13  1040  1041  1042  1043  1044  1045  1046  1047  1048  1049
2020-09-20  1110  1111  1112  1113  1114  1115  1116  1117  1118  1119
2020-09-27  1180  1181  1182  1183  1184  1185  1186  1187  1188  1189
2020-10-04  1250  1251  1252  1253  1254  1255  1256  1257  1258  1259
2020-10-11  1320  1321  1322  1323  1324  1325  1326  1327  1328  1329
2020-10-18  1390  1391  1392  1393  1394  1395  1396  1397  1398  1399
2020-10-25  1460  1461  1462  1463  1464  1465  1466  1467  1468  1469
2020-11-01  1530  1531  1532  1533  1534  1535  1536  1537  1538  1539
2020-11-08  1600  1601  1602  1603  1604  1605  1606  1607  1608  1609
2020-11-15  1670  1671  1672  1673  1674  1675  1676  1677  1678  1679
2020-11-22  1740  1741  1742  1743  1744  1745  1746  1747  1748  1749
2020-11-29  1810  1811  1812  1813  1814  1815  1816  1817  1818  1819
2020-12-06  1880  1881  1882  1883  1884  1885  1886  1887  1888  1889
2020-12-13  1950  1951  1952  1953  1954  1955  1956  1957  1958  1959
2020-12-20  2020  2021  2022  2023  2024  2025  2026  2027  2028  2029
2020-12-27  2090  2091  2092  2093  2094  2095  2096  2097  2098  2099
2021-01-03  2160  2161  2162  2163  2164  2165  2166  2167  2168  2169
2021-01-10  2230  2231  2232  2233  2234  2235  2236  2237  2238  2239
2021-01-17  2300  2301  2302  2303  2304  2305  2306  2307  2308  2309
2021-01-24  2370  2371  2372  2373  2374  2375  2376  2377  2378  2379
2021-01-31  2440  2441  2442  2443  2444  2445  2446  2447  2448  2449
2021-02-07  2510  2511  2512  2513  2514  2515  2516  2517  2518  2519
2021-02-14  2580  2581  2582  2583  2584  2585  2586  2587  2588  2589
2021-02-21  2650  2651  2652  2653  2654  2655  2656  2657  2658  2659
2021-02-28  2720  2721  2722  2723  2724  2725  2726  2727  2728  2729
2021-03-07  2790  2791  2792  2793  2794  2795  2796  2797  2798  2799
2021-03-14  2860  2861  2862  2863  2864  2865  2866  2867  2868  2869
2021-03-21  2930  2931  2932  2933  2934  2935  2936  2937  2938  2939
2021-03-28  3000  3001  3002  3003  3004  3005  3006  3007  3008  3009
2021-04-04  3070  3071  3072  3073  3074  3075  3076  3077  3078  3079
2021-04-11  3140  3141  3142  3143  3144  3145  3146  3147  3148  3149
2021-04-18  3210  3211  3212  3213  3214  3215  3216  3217  3218  3219
2021-04-25  3280  3281  3282  3283  3284  3285  3286  3287  3288  3289
2021-05-02  3350  3351  3352  3353  3354  3355  3356  3357  3358  3359
2021-05-09  3420  3421  3422  3423  3424  3425  3426  3427  3428  3429
2021-05-16  3490  3491  3492  3493  3494  3495  3496  3497  3498  3499
2021-05-23  3560  3561  3562  3563  3564  3565  3566  3567  3568  3569
2021-05-30  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639


# 毎週金曜日

column       c1   c2   c3   c4   c5   c6   c7   c8   c9  c10
index
2020-06-05   40   41   42   43   44   45   46   47   48   49
2020-06-12  110  111  112  113  114  115  116  117  118  119
2020-06-19  180  181  182  183  184  185  186  187  188  189
2020-06-26  250  251  252  253  254  255  256  257  258  259
2020-07-03  320  321  322  323  324  325  326  327  328  329


# 毎時（欠損値は0埋め）

column                 c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 01:00:00     0     0     0     0     0     0     0     0     0     0
2020-06-01 02:00:00     0     0     0     0     0     0     0     0     0     0
2020-06-01 03:00:00     0     0     0     0     0     0     0     0     0     0
2020-06-01 04:00:00     0     0     0     0     0     0     0     0     0     0
...                   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30 20:00:00     0     0     0     0     0     0     0     0     0     0
2021-05-30 21:00:00     0     0     0     0     0     0     0     0     0     0
2021-05-30 22:00:00     0     0     0     0     0     0     0     0     0     0
2021-05-30 23:00:00     0     0     0     0     0     0     0     0     0     0
2021-05-31 00:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649


# 毎時（欠損値は直前の行の値で補間）

column                 c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 01:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 02:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 03:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 04:00:00     0     1     2     3     4     5     6     7     8     9
...                   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30 20:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 21:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 22:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 23:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-31 00:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649

[8737 rows x 10 columns]


column                 c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 01:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 02:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 03:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 04:00:00     0     1     2     3     4     5     6     7     8     9
...                   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30 20:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 21:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 22:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-30 23:00:00  3630  3631  3632  3633  3634  3635  3636  3637  3638  3639
2021-05-31 00:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649

[8737 rows x 10 columns]


# 毎時（欠損値は直後の行の値で補間）

column                 c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 01:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 02:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 03:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 04:00:00    10    11    12    13    14    15    16    17    18    19
...                   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30 20:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 21:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 22:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 23:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-31 00:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649

[8737 rows x 10 columns]


column                 c1    c2    c3    c4    c5    c6    c7    c8    c9   c10
index
2020-06-01 00:00:00     0     1     2     3     4     5     6     7     8     9
2020-06-01 01:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 02:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 03:00:00    10    11    12    13    14    15    16    17    18    19
2020-06-01 04:00:00    10    11    12    13    14    15    16    17    18    19
...                   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
2021-05-30 20:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 21:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 22:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-30 23:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649
2021-05-31 00:00:00  3640  3641  3642  3643  3644  3645  3646  3647  3648  3649


# 毎分（欠損値は前後の値から線形補間）

column                      c1         c2         c3         c4         c5         c6         c7         c8         c9        c10
index
2020-06-01 00:00:00   0.000000   1.000000   2.000000   3.000000   4.000000   5.000000   6.000000   7.000000   8.000000   9.000000
2020-06-01 00:01:00   0.006944   1.006944   2.006944   3.006944   4.006944   5.006944   6.006944   7.006944   8.006944   9.006944
2020-06-01 00:02:00   0.013889   1.013889   2.013889   3.013889   4.013889   5.013889   6.013889   7.013889   8.013889   9.013889
2020-06-01 00:03:00   0.020833   1.020833   2.020833   3.020833   4.020833   5.020833   6.020833   7.020833   8.020833   9.020833
2020-06-01 00:04:00   0.027778   1.027778   2.027778   3.027778   4.027778   5.027778   6.027778   7.027778   8.027778   9.027778
...                        ...        ...        ...        ...        ...        ...        ...        ...        ...        ...
2020-06-01 23:56:00   9.972222  10.972222  11.972222  12.972222  13.972222  14.972222  15.972222  16.972222  17.972222  18.972222
2020-06-01 23:57:00   9.979167  10.979167  11.979167  12.979167  13.979167  14.979167  15.979167  16.979167  17.979167  18.979167
2020-06-01 23:58:00   9.986111  10.986111  11.986111  12.986111  13.986111  14.986111  15.986111  16.986111  17.986111  18.986111
2020-06-01 23:59:00   9.993056  10.993056  11.993056  12.993056  13.993056  14.993056  15.993056  16.993056  17.993056  18.993056
2020-06-02 00:00:00  10.000000  11.000000  12.000000  13.000000  14.000000  15.000000  16.000000  17.000000  18.000000  19.000000
```

<a id="markdown-データ可視化" name="データ可視化"></a>

## データ可視化

<a id="markdown-numpy" name="numpy"></a>

# Numpy

<a id="markdown-scipy" name="scipy"></a>

# Scipy

<a id="markdown-pillowpil-1" name="pillowpil-1"></a>

# Pillow（PIL）

<a id="markdown-scikit-learn-1" name="scikit-learn-1"></a>

# scikit-learn

<a id="markdown-keras-1" name="keras-1"></a>

# Keras

<a id="markdown-tensorflow" name="tensorflow"></a>

# TensorFlow

<a id="markdown-matplotlib-1" name="matplotlib-1"></a>

# Matplotlib

<hr>

Copyright (c) 2020 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

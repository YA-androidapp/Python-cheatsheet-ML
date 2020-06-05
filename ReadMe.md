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
      - [行名・列名、行数・列数・要素数](#行名・列名行数・列数・要素数-2)
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
  - [データ加工](#データ加工-1)
    - [追加](#追加)
      - [行の追加](#行の追加)
        - [append](#append)
        - [concat](#concat)
      - [列の追加](#列の追加)
    - [転置](#転置)
    - [値の置換](#値の置換)
      - [欠損値](#欠損値)
        - [欠損値に置き換える](#欠損値に置き換える)
        - [欠損値を置き換える](#欠損値を置き換える)
          - [定数で置換](#定数で置換)
          - [統計量で置換](#統計量で置換)
          - [前後の要素で置換](#前後の要素で置換)
  - [データ結合](#データ結合)
  - [データ要約](#データ要約)
  - [グループ化](#グループ化)
  - [データ可視化](#データ可視化)

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
```

<a id="markdown-行名・列名行数・列数・要素数-2" name="行名・列名行数・列数・要素数-2"></a>

#### 行名・列名、行数・列数・要素数

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

df4 = pd.concat([df1, df2], join='inner', ignore_index=True)
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

<a id="markdown-データ結合" name="データ結合"></a>

## データ結合

<a id="markdown-データ要約" name="データ要約"></a>

## データ要約

<a id="markdown-グループ化" name="グループ化"></a>

## グループ化

<a id="markdown-データ可視化" name="データ可視化"></a>

## データ可視化

<hr>

Copyright (c) 2020 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

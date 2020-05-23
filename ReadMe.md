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
    - [JSON 文字列の読み込み](#json-文字列の読み込み)
    - [JSON Lines](#json-lines)
    - [Excel 形式（xlsx）](#excel-形式xlsx)
      - [依存パッケージをインストール](#依存パッケージをインストール)
      - [読み込み](#読み込み-3)
      - [書き出し](#書き出し-3)
    - [pickle](#pickle)
      - [読み込み](#読み込み-4)
      - [書き出し](#書き出し-4)
  - [データ抽出](#データ抽出)
  - [データ加工](#データ加工-1)
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

<a id="markdown-python-リストから生成" name="python-リストから生成"></a>

### Python リストから生成

```py
import pandas as pd

lst = range(3) # Pythonのリスト

pd.Series(lst) # 1次元: シリーズ
pd.DataFrame(lst) # 2次元: データフレーム
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

<a id="markdown-データ抽出" name="データ抽出"></a>

## データ抽出

<a id="markdown-データ加工-1" name="データ加工-1"></a>

## データ加工

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

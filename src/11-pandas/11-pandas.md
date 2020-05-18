# pandas

## おまじない

### パッケージの読み込み

```py
import pandas as pd
import numpy as np
```

#### 共通設定

##### 設定値の書き込み

```py
pd.options.display.max_columns = None # 全ての列を表示
pd.set_option('display.max_rows', None) # 全ての行を表示
pd.set_option('max_.*', 100) # 正規表現でも指定できるが、複数該当するとOptionErrorとなる

# その他の設定項目を確認する
import pprint
pprint.pprint(dir(pd.options))

# 既定値に戻す
pd.reset_option('display.max_columns')
pd.reset_option('max_col') # 複数該当した場合はその全てで既定値に戻す
pd.reset_option('^display', silent=True) # display.～をの全てで既定値に戻す
pd.reset_option('all', silent=True)
```

##### 設定値が反映されているか確認

```py
print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_.*')) # 正規表現でも指定できるが、複数該当するとOptionErrorとなる

pd.describe_option() # 全件
pd.describe_option('max.*col') # 正規表現でフィルタリング
```

##### 特定の処理だけ設定変更した状態で実行する

```py
pd.reset_option('display.max_rows')
print(pd.get_option('display.max_rows')) # 60

pd.options.display.max_rows = 10
print(pd.get_option('display.max_rows')) # 10

with pd.option_context('display.max_rows', 3): # withブロックの中だけ有効
    print(pd.get_option('display.max_rows')) # 3

print(pd.get_option('display.max_rows')) # 10
```

## データ型

## I/O

## データ抽出

## データ加工

## データ結合

## データ要約

## グループ化

## データ可視化

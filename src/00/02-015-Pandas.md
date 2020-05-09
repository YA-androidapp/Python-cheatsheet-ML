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

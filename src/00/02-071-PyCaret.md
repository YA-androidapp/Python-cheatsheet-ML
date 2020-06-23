## PyCaret

（Windows）

```powershell
$ python -m pip install -U pip
$ python -m venv myenv
$ .\myenv\Scripts\activate
(myenv)$ python -m pip install pycaret

(myenv)$ python -c "from pycaret.utils import version;version()" # 動作確認
```

```

```

---

（Mac）

```sh
# Macでは別途LightGBMをインストールする必要がある
$ brew install cmake
$ brew install libomp
$ git clone --recursive https://github.com/microsoft/LightGBM ; cd LightGBM
$ mkdir build ; cd build
$ cmake ..
$ make -j4

$ python -m pip install -U pip
$ python -m venv myenv
$ source ./myenv/bin/activate
(myenv)$ python -m pip install pycaret

(myenv)$ python -c "from pycaret.utils import version;version()" # 動作確認
```

```
1.0.0
```

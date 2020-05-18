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

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

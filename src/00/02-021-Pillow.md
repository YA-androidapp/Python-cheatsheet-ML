## Pillow（PIL）

（Windows）

```ps
$ python -m pip install -U pip
$ python -m venv myenv
$ .\numpyenv\Scripts\activate
(numpyenv)$ python -m pip install --user pillow # NumPyもインストールされる

$ python -c "import PIL" # エラーが出なければ正常にインストールされている
```

※パッケージ名は Pillow に改名されているが、import 時には PIL を指定する必要がある

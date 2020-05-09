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

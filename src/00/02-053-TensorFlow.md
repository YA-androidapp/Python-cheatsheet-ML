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

### CUDA のインストール

- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) から、CUDA Toolkit 10.1 のインストーラーを取得する
  - [Installer for Windows 10 x86_64](http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe)

### cuDNN のインストール

- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
  - [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)
    - [cuDNN Library for Windows 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip)

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

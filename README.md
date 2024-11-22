# bbocr-v2

Paper

**bbOCR: An Open-source Multi-domain OCR Pipeline for Bengali Documents**

https://arxiv.org/abs/2308.10647



# CPU Installation 
* create conda environment and activate

```shell
conda create -n bbocrv2 python=3.9
conda activate bbocrv2
```

* install requirements

```shell
pip install apsisocr
pip install onnxruntime
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install pycocotools
pip install scikit-learn
pip install shapely
pip install numpy==1.23.0
```

# GPU Installation 
* create conda environment and activate

```shell
conda create -n bbocrv2gpu python=3.9
conda activate bbocrv2gpu
```

* installing cudatoolkit and cudnn:

```shell
conda install cudatoolkit
conda install cudnn
```

* install dependencies

```shell
pip install apsisocr
pip install onnxruntime-gpu==1.16.0
python -m pip install -U fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install shapely
pip install pycocotools
pip install scikit-learn
pip install numpy==1.23.0
```

* export LD_LIBRARY_PATH [**Linux** and **MAC**]

```shell 
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
Above didn't work, this worked for me (Reasat)
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_conda_env>/lib/python3.9/site-packages/torch/lib 
```

* LD_LIBRARY_PATH setting in **Windows**:[From ChatGPT- Needs Testing]

```shell
mkdir %CONDA_PREFIX%\etc\conda\activate.d
echo set "LD_LIBRARY_PATH=%CUDNN_PATH%\lib;%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH%" >> %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
```

# GPU Testing Note

The first Load run will take more time 

```shell
Execution Time for Layout Prediction and Text Recognition: 3.7 seconds
```
The second run on the same data will take less time 

```shell
Execution Time for Layout Prediction and Text Recognition: 0.73 seconds
```

## GPU Testing Environment Info

```shell

-----------------------------::neofetch::---------------------------------
            .-/+oossssoo+/-.               ansary@ML-PROD 
        `:+ssssssssssssssssss+:`           -------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 20.04.6 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: Precision 7920 Rack 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 5.15.0-105-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 10 days, 19 hours, 7 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 2325 (dpkg), 13 (snap) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.0.17 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Terminal: node 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: Intel Xeon Silver 4214R (24) @ 3.500GHz 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   GPU: NVIDIA 65:00.0 NVIDIA Corporation Device 2230 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Memory: 21602MiB / 63915MiB 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/                            
  +sssssssssdmydMMMMMMMMddddyssssssss+                             
   /ssssssssssshdmNNNNmyNMMMMhssssss/
    .ossssssssssssssssssdMMMNysssso.
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.


---------------------::nvidia-smi::----------------------------------------------------

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A6000               Off | 00000000:65:00.0 Off |                  Off |
| 30%   30C    P8              20W / 300W |  10036MiB / 49140MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

```

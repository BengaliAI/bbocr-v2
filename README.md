# bbocr-v2


# Directory Setup

Jumpstart? How?

- Create a folder called "reconstruction"
- Create another folder called "templates" inside reconstruction
- Populate with appropriate HTML files (kinda vague, but can't wrap everything in this notebook now. Just ask me aka Istiak Shihab). Also I will provide you a snapshot of my current directory structure. Just copy over the files needed. I guess.
- Create another folder called "img_src" inside reconstruction
- Create another folder called "image" inside reconstruction
- Create another folder called "html_output" inside reconstruction
- Now get out of reconstruction folder and Create YET another folder called "image". This is where you will keep your PNG images to run inference on.
- Make sure the folder structure is like this: image/   best.pt    make-html.ipynb  environment.yml    reconstruction/...
- Install Based on device
- You should be good to go? Hopefully.
- Good Luck!

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
pip install shapely
```
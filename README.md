# RTFNet-pytorch

This is the official pytorch implementation of [RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes](https://github.com/yuxiangsun/RTFNet/blob/master/doc/RAL2019_RTFNet.pdf) (IEEE RAL). The util, train, test and demo codes are heavily borrowed from [MFNet](https://github.com/haqishen/MFNet-pytorch). 

Note that our implementations of the evaluation metrics (Acc and IoU) are different from those in MFNet. In addition, we consider the unlabelled class when computing the metrics. We think that it is fine to directly import our results (including the compared networks) in your paper if you use our `test.py` to evaluate your model.

The current version supports Python 3.6, CUDA 10.1 and PyTorch 1.1.

<img src="doc/network.png" width="900px"/>
  
## Introduction

RTFNet is a data-fusion network for semantic segmentation. It consists of two encoders and one decoder. Although RTFNet is designed with RGB-Thermal data, it generalizes well for RGB-D data. Please take a look at this IEEE RAL 2019 [paper](https://doi.org/10.1109/LRA.2019.2932874).
 
## Dataset
 
The original dataset can be downloaded from the MFNet project [page](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), but you are encouraged to download our preprocessed dataset from [here](http://gofile.me/4jm56/CfukComo1).

## Pretrained weights

The weights used in the paper:

RTFNet 50: http://gofile.me/4jm56/9VygmBgPR
RTFNet 152: http://gofile.me/4jm56/ODE2fxJKG

## Usage

* Assume you have nvidia docker installed. To reproduce our results:
```
$ cd ~ 
$ git clone https://github.com/yuxiangsun/RTFNet.git
$ mkdir ~/RTFNet/dataset
$ cd ~/RTFNet/dataset
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
$ mkdir -p ~/RTFNet/weights_backup/RTFNet_50
$ cd ~/RTFNet/weights_backup/RTFNet_50
$ (download the RTFNet_50 weight in this folder)
$ mkdir -p ~/RTFNet/weights_backup/RTFNet_152
$ cd ~/RTFNet/weights_backup/RTFNet_152
$ (download the RTFNet_152 weight in this folder)
$ docker build -t rtfnet_docker_image .
$ nvidia-docker run -it --shm-size 8G -p 1234:6006 --name rtfnet_docker -v ~/RTFNet:/opt/project rtfnet_docker_image
$ (currently, you should be in the docker)
$ cd /opt/project 
$ python3 test.py
$ python3 run_demo.py
```

* To train RTFNet (please mannully change RTFNet variants in the model file):
```
$ cd ~ 
$ git clone https://github.com/yuxiangsun/RTFNet.git
$ mkdir ~/RTFNet/dataset
$ cd ~/RTFNet/dataset
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
$ docker build -t rtfnet_docker_image .
$ nvidia-docker run -it --shm-size 8G -p 1234:6006 --name rtfnet_docker -v ~/RTFNet:/opt/project rtfnet_docker_image
$ (currently, you should be in the docker)
$ cd /opt/project 
$ python3 train.py
$ (fire up another terminal)
$ nvidia-docker exec -it rtfnet_docker bash
$ cd /opt/project/runs
$ tensorboard --logdir=.
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```

## Citation

If you use RTFNet in an academic work, please cite:

```
@ARTICLE{sun2019rtfnet,
author={Yuxiang Sun and Weixun Zuo and Ming Liu}, 
journal={{IEEE Robotics and Automation Letters}}, 
title={{RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes}}, 
year={2019}, 
volume={4}, 
number={3}, 
pages={2576-2583}, 
doi={10.1109/LRA.2019.2904733}, 
ISSN={2377-3766}, 
month={July},}
```

## Demos

<img src="doc/demo.png" width="900px"/>

## Contact

sun.yuxiang@outlook.com, http://eeyxsun.people.ust.hk/


# RTFNet-pytorch

RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes

<img src="doc/network.png" width="900px"/>
 
This is the official pytorch implementation of [RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes](http://eeyxsun.people.ust.hk/docs/RAL2019_rtfnet.pdf) (IEEE RAL). The util, test and demo codes are heavily borrowed from [MFNet](https://github.com/haqishen/MFNet-pytorch). Note that we re-implemented the evaluation metrics (Acc and IoU), which are different from those in MFNet. In addition, we consider the unlabelled class when computing the metrics.
 
## Introduction

RTFNet is a data-fusion network for semantic segmentation. It consists of two encoders and one decoder. 
 
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
$ cd ~/RTFNet/dataset
$ (download our preprocessed dataset.zip in this folder)
$ unzip -d .. dataset.zip
$ cd ~/RTFNet/weights_backup/RTFNet_50
$ (download the RTFNet_50 weight in this folder)
$ cd ~/RTFNet/weights_backup/RTFNet_152
$ (download the RTFNet_152 weight in this folder)
$ docker build -t rtfnet_docker_image .
$ nvidia-docker run -it --shm-size 8G --name rtfnet_docker -v ~/RTFNet_PyTorch:/opt/project rtfnet_docker_image
$ (currently, you should be in the docker)
$ cd /opt/project 
$ python test.py
$ python run_demo.py
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
sun.yuxiang@outlook.com

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y vim python3 python3-pip 

RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=40.3.0 

RUN pip3 install -U scipy scikit-learn
RUN pip3 install torch>=1.7 torchvision torchsummary
RUN pip3 install tensorboard==2.2.0
#RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=1.7 torchvision # for China mainland users

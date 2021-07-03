FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y vim python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=40.3.0 

RUN pip3 install -U scipy scikit-learn
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchsummary==1.5.1
RUN pip3 install tensorboard==2.2.0

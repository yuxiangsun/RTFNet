FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y vim python3 python3-pip 

RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=40.3.0 

RUN pip3 install -U scipy scikit-learn
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchsummary
RUN pip3 install tensorboard==2.2.0

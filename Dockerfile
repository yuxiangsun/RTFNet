FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 
RUN apt-get update && apt-get install -y vim python3 python3-pip 
RUN pip3 install -U scipy==1.3.0 scikit-learn==0.21.3
RUN pip3 install tensorboardX==1.8 torchsummary==1.5.1 tensorflow==1.14.0
RUN pip3 install torch==1.2 torchvision==0.4.0
#RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.2 torchvision # for China mainland users

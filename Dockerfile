FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04 
RUN apt-get update && apt-get install -y vim python python-pip 
RUN pip install tensorboardX==1.7 torchsummary==1.5.1 
RUN pip install -U scipy==1.2.1 scikit-learn==0.20.3  
RUN pip install numpy==1.15.4
RUN pip install torch==0.4.1 torchvision==0.2.2 
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==0.4.1 torchvision==0.2.2 # for China mainland users


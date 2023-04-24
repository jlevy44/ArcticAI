FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

WORKDIR /arcticai

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get -y update
RUN apt-get -y install libgdal-dev
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-opencv
RUN apt-get -y install ffmpeg libsm6 libxext6

# install gcc
RUN apt-get install build-essential

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# install detectron2 and other dependencies
RUN python3 -m pip install numpy
RUN python3 -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels
RUN python3 -m pip install fire
RUN python3 -m pip install opencv-python
RUN python3 -m pip install scikit-image
RUN python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
COPY install_scripts/* /arcticai
COPY requirements.txt /arcticai
COPY additional_requirements.txt /arcticai
RUN sh install.sh
RUN pip install -r requirements.txt
RUN pip install -r additional_requirements.txt
RUN pip install arctic-ai
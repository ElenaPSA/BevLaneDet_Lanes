FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8

# Compilation of MMCV
RUN conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install --upgrade pip wheel setuptools
RUN pip install torchpack
RUN git clone --branch v1.4.0 https://github.com/open-mmlab/mmcv.git 
WORKDIR /mmcv
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py develop

# Python packages for all the models
WORKDIR /
COPY requirements_annot3d.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_annot3d.txt

# Setup of bevfusion
RUN mkdir /bevfusion
#COPY ./bevfusion/setup.py /bevfusion/
COPY setup.py /bevfusion/
#COPY ./bevfusion/mmdet3d /bevfusion/mmdet3d
COPY mmdet3d /bevfusion/mmdet3d
WORKDIR /bevfusion/
RUN FORCE_CUDA=1 python setup.py develop

# Setup of MS3D
WORKDIR /
COPY ./MS3D /MS3D/
WORKDIR /MS3D/
RUN FORCE_CUDA=1 python setup.py develop
WORKDIR /MS3D/tracker/
RUN FORCE_CUDA=1 python setup.py develop

RUN apt-get update && apt-get install cmake libopencv-dev -y

# Setup of trackingtools
WORKDIR /
COPY ./trackingtools /trackingtools/
WORKDIR /trackingtools/
RUN python setup.py develop


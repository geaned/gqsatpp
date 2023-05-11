# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN apt-get update -y && apt-get install -y software-properties-common curl sudo gcc unzip
RUN add-apt-repository -y ppa:jblgf0/python
RUN apt-get update -y &&  add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils vim man build-essential python3.6 python3.6-dev wget htop zlib1g-dev swig&& \
    rm -rf /var/lib/apt/lists/*

#RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.7
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py | sudo python3.6

RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX
RUN pip3 install torch==1.1.0 torchvision
RUN pip3 install gym setuptools

ENV CUDA_HOME /usr/local/cuda
ENV CPATH /usr/local/cuda/include
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/home/gqsat_user/gqsat/minisat/build/release/lib:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 7.5+PTX"

RUN pip3 install --verbose --no-cache-dir torch_scatter==1.3.1
RUN pip3 install --verbose --no-cache-dir torch-sparse==0.4.0
RUN pip3 install --verbose --no-cache-dir torch-cluster==1.4.4
RUN pip3 install torch-geometric==1.3.1

RUN pip3 install tqdm gdown

RUN useradd -ms /bin/bash satgym_user
USER satgym_user
WORKDIR /home/satgym_user/gym/
RUN echo "alias python=/usr/bin/python3.6" >> /home/satgym_user/.bashrc

# to run the minisat binary
ENV PATH /home/satgym_user/minisat/build/release/bin:$PATH

RUN echo 'just run [nvidia-docker run -p 8807:8807 -v "`pwd`:/home/satgym_user/gym/" -it gym:latest bash]'
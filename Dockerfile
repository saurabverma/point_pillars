# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# apt tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        less \
        fish \
        libsparsehash-dev \
        && \
# ==================================================================
# python and pip
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL setuptools && \
    $PIP_INSTALL cmake && \
    $PIP_INSTALL numpy && \
    $PIP_INSTALL scipy && \
    $PIP_INSTALL matplotlib && \
    $PIP_INSTALL Cython && \
    $PIP_INSTALL https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl && \
    $PIP_INSTALL https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl && \
    $PIP_INSTALL shapely && \
    $PIP_INSTALL fire && \
    $PIP_INSTALL tqdm && \
    $PIP_INSTALL pybind11 && \
    $PIP_INSTALL tensorboardX && \
    $PIP_INSTALL protobuf && \
    $PIP_INSTALL scikit-image && \
    $PIP_INSTALL numba && \
    $PIP_INSTALL pillow && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# setup
# ------------------------------------------------------------------
RUN echo 'force_color_prompt=yes' >> /root/.bashrc && \
    mkdir /root/point_pillars
WORKDIR /root/point_pillars

RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz

# RUN git clone https://github.com/saurabverma/second.pytorch.git --depth 1
RUN git clone https://github.com/traveller59/SparseConvNet.git --depth 1 && \
    cd ./SparseConvNet && \
    python setup.py install && \
    cd .. && \
    rm -rf SparseConvNet

ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice

# ENV PYTHONPATH=/root/point_pillars/second.pytorch
# WORKDIR /root/point_pillars/second.pytorch/second

ENTRYPOINT ["fish"]

# # NOTE: The following folders are assumed to be already setup from the host PC
# /root/point_pillars/data --> holds the KIITI or other data (see README.md for details)
# /root/point_pillars/model --> holds the trained model checkpoints
# /root/point_pillars/src/second.torch --> actual pointpillars code

# # Build image using (inside the directory with Dockerfile):
# docker build --tag="point_pillars:Dockerfile" .

# # Create container using (to be auto-removed on stop):
# # NOTE: See 'note' file for specific details
# docker run -d -it --rm --runtime=nvidia -v /home/docker_mount/<specific_folder>/point_pillars:/root/point_pillars --net=host --name pointpillars_test point_pillars:Dockerfile

# # Run container using (in multiple terminals, as needed):
# docker exec -it pointpillars_test bash

# # Stop container using:
# docker stop pointpillars_test
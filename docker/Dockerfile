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
        apt-utils \
        wget \
        git \
        vim \
        nano \
        less \
        fish \
        cmake \
        libsparsehash-dev \
        libopenni-dev \
        libproj-dev \
        libpcl-dev \
        && \
# ==================================================================
# python and pip
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
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
    $PIP_INSTALL pandas && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so && \
    ln -s /usr/lib/python2.7/dist-packages/vtk/libvtkRenderingPythonTkWidgets.x86_64-linux-gnu.so /usr/lib/x86_64-linux-gnu/libvtkRenderingPythonTkWidgets.so && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* && \
# ==================================================================
# setup
# ------------------------------------------------------------------
    echo 'force_color_prompt=yes' >> /root/.bashrc && \
    mkdir /root/point_pillars && \
    cd /root/point_pillars && \
    git clone https://github.com/traveller59/SparseConvNet.git --depth 1 && \
    cd ./SparseConvNet && \
    python setup.py install && \
    cd .. && \
    rm -rf SparseConvNet

ENV KITTI_DATASET_ROOT=/root/point_pillars/data/kitti_second \
    MODEL_ROOT=/root/point_pillars/model \
    PYTHONPATH=/root/point_pillars/src/point_pillars

WORKDIR /root/point_pillars

# # git repo to be set from outside docker such that user can easily manipulate
# WORKDIR /root/point_pillars/src/
# RUN git clone https://github.com/saurabverma/point_pillars.git --depth 1
# WORKDIR /root/point_pillars/src/point_pillars/second

# # NUMPRO is deprecated and thus, not needed
# ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
# ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
# ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
# ENV PYTHONPATH=/root/point_pillars/src/point_pillars

ENTRYPOINT ["fish"]

# # NOTE: The following folders are assumed to be already setup from the host PC
# /root/point_pillars/data --> holds the KIITI or other data (see README.md for details)
# /root/point_pillars/model --> holds the trained model checkpoints
# /root/point_pillars/src/point_pillars --> actual pointpillars code

# # Build image using (inside the directory with Dockerfile):
# docker build --tag="point_pillars:Dockerfile" .

# # Create container using (to be auto-removed on stop):
# # NOTE: See 'note' file for specific details
# docker run -d -it --rm --runtime=nvidia -v /home/docker_mount/<parent_folder_to_point_pillars>:/root/point_pillars --net=host --name pointpillars_test point_pillars:Dockerfile

# # Run container using (in multiple terminals, as needed):
# docker exec -it pointpillars_test fish

# # Stop container using:
# docker stop pointpillars_test
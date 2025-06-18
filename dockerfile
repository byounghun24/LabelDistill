# Base: CUDA 11.1 + Ubuntu 20.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 비대화 설치 + 타임존 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    bash sudo wget bzip2 git curl ca-certificates libx11-6 libgl1 libglib2.0-0 tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 사용자 추가
ARG USERNAME=byounghun
ARG USER_UID=1008
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# Miniconda 설치
ENV CONDA_DIR=/home/$USERNAME/miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda clean -a -y

# Conda 환경 생성
RUN conda create -n labeldistill python=3.8 -y
SHELL ["/bin/bash", "-c"]
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate labeldistill" >> ~/.bashrc

# PyTorch (v1.9.0 + cu111)
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate labeldistill && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Clone LabelDistill repo
RUN git clone https://github.com/byounghun24/LabelDistill.git
WORKDIR /home/$USERNAME/LabelDistill

# Install mmcv, mmdet, mmseg
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate labeldistill && \
    pip install openmim && \
    mim install mmcv-full==1.6.0 && \
    mim install mmdet==2.26.0 && \
    mim install mmsegmentation==0.29.1

# Install mmdetection3d
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate labeldistill && \
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && pip install -e .

# Install pip
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate labeldistill && \
    pip install pip==24.0

# Install project dependencies
RUN source $CONDA_DIR/etc/profile.d/conda.sh && conda activate labeldistill && \
    pip install -r requirements.txt && \
    python setup.py develop

CMD ["bash"]

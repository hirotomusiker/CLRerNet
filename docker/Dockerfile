FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo git zip curl \
    libopencv-dev libffi-dev liblapack-dev libsqlite3-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev \
    ssh-client wget vim python3-opencv &&\
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# user setting
ARG UID
RUN useradd docker -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker

# python
ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ARG PYTHON_VERSION
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# torch
ARG TORCH_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG TORCHVISION_VERSION
RUN pip install "numpy<2.0"
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchvision==${TORCHVISION_VERSION}

# mmcv
ARG MIM_VERSION
ARG MMCV_VERSION
ARG MMENGINE_VERSION
ARG MMDET_VERSION
RUN pip install -U openmim==${MIM_VERSION}
RUN mim install mmcv==${MMCV_VERSION}
RUN mim install mmengine==${MMENGINE_VERSION}
RUN mim install mmdet==${MMDET_VERSION}

# requirements
RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# lane NMS
COPY --chown=docker libs/models/layers/nms /tmp/nms
WORKDIR /tmp/nms
RUN python /tmp/nms/setup.py install

# enable deterministic training
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# path
ENV PYTHONPATH $PYTHONPATH:/work
ENV PYTHONPATH $PYTHONPATH:/home/docker/

WORKDIR /work

ENTRYPOINT ["/bin/bash"]

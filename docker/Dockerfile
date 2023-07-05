FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
ARG TORCHVISION_VERSION
RUN pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==${TORCHVISION_VERSION}

# mmcv
ARG MIM_VERSION
ARG MMCV_VERSION
RUN pip install -U openmim==${MIM_VERSION}
RUN mim install mmcv-full==${MMCV_VERSION}

# mmdet
ARG MMDET_VERSION
WORKDIR /home/docker/
RUN pip install Cython
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection -b v${MMDET_VERSION} --depth 1
WORKDIR mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .

# requirements
RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# lane NMS
COPY --chown=docker libs/models/layers/nms /tmp/nms
WORKDIR /tmp/nms
RUN python /tmp/nms/setup.py install

# path
ENV PYTHONPATH $PYTHONPATH:/work
ENV PYTHONPATH $PYTHONPATH:/home/docker/

WORKDIR /work

ENTRYPOINT ["/bin/bash"]

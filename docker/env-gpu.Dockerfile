FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y \
    git \
    htop \
    openssh-server \
    software-properties-common \
    vim \
    unzip

WORKDIR /home/dev/src
COPY ./src/requirements.txt ./requirements.txt

RUN pip --no-cache-dir install --upgrade pip \
    && pip --no-cache-dir install -r requirements.txt

CMD ["bash"]
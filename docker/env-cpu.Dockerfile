FROM python:3.7.11
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y \
    git \
    htop \
    openssh-server \
    software-properties-common \
    vim \
    unzip

RUN pip --no-cache-dir install --upgrade \
    pip \
    torch==1.10.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

WORKDIR /home/dev/src
COPY ./src/requirements.txt ./requirements.txt

RUN pip --no-cache-dir install --upgrade pip \
    && pip --no-cache-dir install -r requirements.txt

CMD ["bash"]
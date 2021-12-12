FROM python:3.7.11
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip --no-cache-dir install --upgrade \
    pip \
    pandas

WORKDIR /home/dev

CMD ["scripts/setup/data"]
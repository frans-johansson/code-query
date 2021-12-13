FROM python:3.7.11
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /home/dev

CMD ["scripts/setup/data"]
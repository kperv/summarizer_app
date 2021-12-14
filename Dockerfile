FROM ubuntu:18.04

RUN apt update
RUN apt install -y build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git

WORKDIR /sum_app

COPY ./requirements.txt /sum_app/requirements.txt
COPY ./clustering_model.py /sum_app/clustering_model.py

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /sum_app


s
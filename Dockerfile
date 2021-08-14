FROM ubuntu:18.04

RUN apt update
RUN apt install -y build-essential \
  python3 \
  python3-pip \
  python3-dev \
  git

RUN python3 -m pip install jupyter \
  transformers[torch]

FROM ubuntu:18.04

RUN apt update
RUN apt install -y build-essential \
  python3 \
  python3-pip \
  python3-dev \
  git

RUN python3 -m pip install flask \
  transformers[sentencepiece] \
  -U bert_score \
  https://huggingface.co/spacy/ru_core_news_md/resolve/main/ru_core_news_md-any-py3-none-any.whl \
  https://huggingface.co/spacy/es_core_news_md/resolve/main/es_core_news_md-any-py3-none-any.whl

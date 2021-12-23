#FROM huggingface/transformers-pytorch-cpu:latest
FROM ubuntu:20.04



#RUN python3 -m pip install --upgrade pip && pip3 install pandas \ 
                 #nltk \
                 #sklearn \
                 #datasets \
                 #torch \
                 #transformers \
                 #numpy \
                 #rouge==0.3.1 \
                 #flask

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y build-essential \
    python3 \
    python3-pip \
    python3-dev && \ 
    pip3 install pandas \ 
                 nltk \
                 sklearn \
                 datasets \
                 torch \
                 transformers \
                 numpy \
                 rouge==0.3.1 \
                 flask

WORKDIR /sum_app
    
#COPY requirements.txt /requirements.txt



#RUN pip3 install --no-cache-dir -r /sum_app/requirements.txt


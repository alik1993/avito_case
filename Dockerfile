# FROM python:3.8.6-buster

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils && \
    apt-get install zip unzip && \
    apt-get install -y p7zip-full && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

# RUN cat distilbert_avito/pytorch_model.z01 distilbert_avito/pytorch_model.z02 distilbert_avito/pytorch_model.z03 distilbert_avito/pytorch_model.z04 distilbert_avito/pytorch_model.z05 distilbert_avito/pytorch_model.z06 distilbert_avito/pytorch_model.zip  > distilbert_avito/combined.zip

# RUN unzip -FF distilbert_avito/combined.zip -d distilbert_avito/
 
RUN 7z x distilbert_avito/pytorch_model.z01  -odistilbert_avito
RUN ls distilbert_avito

RUN pip install -r requirements.txt

CMD python lib/run.py

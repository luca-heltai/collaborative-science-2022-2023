FROM ubuntu:focal

RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && pip install numpy \
    && pip install --upgrade MDAnalysis

RUN mkdir Trial

WORKDIR \Trial
COPY hello_world.py hello_world.py
RUN python3 hello_world.py
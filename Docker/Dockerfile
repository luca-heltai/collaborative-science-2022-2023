FROM ubuntu:focal

RUN cd && \
    apt-get update && apt-get install -y vim wget && \
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
    sh Anaconda3-2022.10-Linux-x86_64.sh -b -p /anaconda 
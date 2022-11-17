FROM ubuntu:focal

RUN apt-get update  && apt-get install -y vim wget  python3-pip
    

ENV CONDA_DIR /opt/conda

RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
    sh Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/conda 

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia && \
    conda install -y -c conda-forge tensorflow


    




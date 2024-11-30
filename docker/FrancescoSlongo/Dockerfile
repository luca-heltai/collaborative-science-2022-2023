FROM ubuntu:focal

RUN apt-get update && \
    apt-get install -y vim wget && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y --override-channels -c conda-forge numpy matplotlib mdanalysis && \
    conda update -y numpy matplotlib mdanalysis 

WORKDIR /home

COPY hello_world.py hello_world.py

RUN python3 hello_world.py

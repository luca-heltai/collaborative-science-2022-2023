FROM ubuntu:focal

RUN apt-get update && \
    apt-get install -y vim wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y -c conda-forge pyomo && \
	conda install -y -c conda-forge glpk && \
	conda install -y numpy && \
	conda install pandas && \
    conda install -y -c anaconda pytest && \
    conda activate base

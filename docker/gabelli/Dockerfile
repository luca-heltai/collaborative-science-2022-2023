FROM ubuntu:focal

RUN cd && \
    apt-get update && \
    apt-get install -y vim wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y -c conda-forge pyomo && \
	conda install -y -c conda-forge glpk && \
	conda install -y numpy && \
	conda install pandas

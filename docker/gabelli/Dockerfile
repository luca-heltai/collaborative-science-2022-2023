FROM ubuntu:focal

RUN cd && \
    apt-get update && \
    apt-get install -y vim wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
    sh Anaconda3-2022.10-Linux-x86_64.sh -b -p /anaconda 

RUN conda install -y -c conda-forge pyomo && \
	conda install -y -c conda-forge glpk && \
	conda install -y numpy && \
	conda install pandas

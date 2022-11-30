#FROM ubuntu:latest
FROM ubuntu:focal

RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && pip install numpy \
    && pip install matplotlib
RUN echo 'echo "This image runs Ubuntu and has vim, python, pip, numpy and matplotlib installed."' > /usr/local/bin/start.sh
RUN echo 'echo "For a simple test, run \"python3 main.py\"."' >> /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /home

ENTRYPOINT /usr/local/bin/start.sh && /bin/bash
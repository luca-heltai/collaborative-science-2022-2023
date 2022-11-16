FROM ubuntu:focal

RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && pip install numpy \
    && pip install --upgrade "jax[cpu]"
RUN echo 'echo "This image runs Ubuntu and has vim, python, pip, numpy and jax installed."' > /usr/local/bin/start.sh
RUN echo 'echo "For a simple test, run \"python3 test.py\"."' >> /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /home

ENTRYPOINT /usr/local/bin/start.sh && /bin/bash

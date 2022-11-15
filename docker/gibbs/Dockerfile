#FROM ubuntu:latest
FROM python:3.10




WORKDIR /root
RUN mkdir gibbs_project


ADD EpsiGreedyMultiArme.py  FairnessTrial.py  FourthPhase.py  InitializationAndFirstPhase.py main.py  SecondPhasetrial.py ThirdPhase.py /root/gibbs_project/

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install numpy matplotlib 

CMD ["python", "./gibbs_project/main.py"]
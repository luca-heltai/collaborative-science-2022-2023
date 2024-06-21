FROM ubuntu:latest

RUN apt-get update

RUN apt install g++ -y

COPY test_cpp.cpp .

RUN g++ test_cpp.cpp

RUN ./*.out




FROM ubuntu:22.04

RUN apt-get update 

RUN apt-get install -y wget 

RUN apt-get install -y git

RUN apt install -y python3 python3-pip

RUN mkdir -p /miniconda3

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh

RUN bash /miniconda3/miniconda.sh -b -u -p /miniconda3

RUN rm -rf /miniconda3/miniconda.sh

RUN /miniconda3/bin/conda init bash

WORKDIR /app

COPY ./environment.yml /app/environment.yml

RUN /miniconda3/bin/conda env create -v --file environment.yml

SHELL ["/miniconda3/bin/conda", "activate", "PdM-Evaluation", "/bin/bash", "-c"]

COPY . /app

WORKDIR /app/src/pdm-evaluation

ENTRYPOINT ["/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "PdM-Evaluation", "python3"]
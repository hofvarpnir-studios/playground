FROM --platform=linux/amd64 continuumio/miniconda3
SHELL ["conda", "run", "/bin/bash", "-c"]
# Install system dependencies
RUN apt update && apt install -y git strace curl vim g++ && rm -rf /var/lib/apt/lists/*
# Set CXX env var 
ENV CXX g++
# Install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.1.0-beta.1/s5cmd_2.1.0-beta.1_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin && s5cmd --help
# Install pytorch
RUN conda install python=3.10 pytorch torchvision pytorch-cuda=11.7 -c pytorch-nightly -c nvidia && conda clean -a -y
# Install python dependencies
COPY . /code/playground
WORKDIR /code/playground
RUN pip install -r playground/requirements.txt
RUN pip install -e .
# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
WORKDIR /code/
ENTRYPOINT ["/usr/bin/tini", "--"]
FROM --platform=linux/amd64 continuumio/miniconda3
SHELL ["conda", "run", "/bin/bash", "-c"]
# Install system dependencies
RUN apt update && apt install -y git strace vim && rm -rf /var/lib/apt/lists/*
# Install base python dependencies
RUN conda install python=3.10 pytorch torchvision pytorch-cuda=11.7 -c pytorch-nightly -c nvidia && conda clean -a -y
# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt; rm /tmp/requirements.txt
# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["conda", "run", "/usr/bin/tini", "--"]
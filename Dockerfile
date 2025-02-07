FROM tensorflow/tensorflow:2.13.0

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Preparing the model files
RUN apt-get update && apt-get -y install unzip curl
COPY brainmask.zip /data/model/
WORKDIR /data/model
RUN unzip brainmask.zip && rm brainmask.zip

# Installing dcm2niix
WORKDIR /usr/local/bin
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip && \
    unzip dcm2niix_lnx.zip  && \
    rm dcm2niix_lnx.zip

# Installing the defacer package
COPY src /usr/local/defacer/src
COPY pyproject.toml /usr/local/defacer/

WORKDIR /usr/local/defacer
RUN pip install . && pip cache purge
CMD ["deface", "--help"]
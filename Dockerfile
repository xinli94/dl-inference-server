FROM ubuntu:16.04

# Build for Python 2.7 (options are 2.7 or 3.5)
ARG PYVER=2.7

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common && \
    add-apt-repository -y ppa:maarten-fonville/protobuf && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            build-essential \
            curl \
            libcurl3-dev \
            libopencv-dev \
            libopencv-core-dev \
            libprotobuf-dev \
            protobuf-compiler \
            python$PYVER \
            python$PYVER-dev && \
    ldconfig

# Make /usr/bin/python point to the $PYVER version of python
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python`echo $PYVER | cut -c1-1` && \
    ln -s /usr/bin/python$PYVER /usr/bin/python && \
    ln -s /usr/bin/python$PYVER /usr/bin/python`echo $PYVER | cut -c1-1`

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python$PYVER get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade setuptools

WORKDIR /workspace
COPY . .
RUN make -j4 -f Makefile.clients all pip

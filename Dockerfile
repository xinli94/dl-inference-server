# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM ubuntu:16.04

# Build for Python 2.7 (options are 2.7 or 3.5)
ARG PYVER=2.7

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common && \
    add-apt-repository -y ppa:maarten-fonville/protobuf && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            autoconf \
            automake \
            build-essential \
            clang \
            curl \
            git \
            libc++-dev \
            libcurl3-dev \
            libgflags-dev \
            libgtest-dev \
            libopencv-dev \
            libopencv-core-dev \
            libprotobuf-dev \
            libtool \
            pkg-config \
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

RUN pip install --upgrade setuptools grpcio-tools

# Clients and examples require gRPC
WORKDIR /opt
RUN git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc && \
    cd grpc && \
    git submodule update --checkout --init && \
    make -j"$(nproc)" && make -j"$(nproc)" install

WORKDIR /workspace
COPY . .
RUN make -j4 -f Makefile.clients all pip

FROM docker.io/intel/intel-optimized-pytorch:2.3.110-xpu-pip-base as base

RUN apt-get update && apt-get install -y \
    clang cmake lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN pip install intel-npu-acceleration-library==1.4.0 \
                intel_extension_for_pytorch==2.5.0

FROM base as drivers

RUN pip install optimum[openvino]==1.23.3

WORKDIR /drivers

RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-driver-compiler-npu_1.10.0.20241107-11729849322_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-fw-npu_1.10.0.20241107-11729849322_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-level-zero-npu_1.10.0.20241107-11729849322_ubuntu22.04_amd64.deb && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.44/level-zero_1.17.44+u22.04_amd64.deb && \
    apt-get update && apt-get install -y libtbb12 && \
    dpkg -i *.deb && \
    rm -rf /var/lib/apt/lists/*

FROM drivers

WORKDIR /src/dictation

COPY requirements.txt /src/dictation/requirements.txt

RUN pip install -r requirements.txt

COPY . /src/dictation

WORKDIR /src/dictation

# Models will be mounted from user's home directory
VOLUME /root/.whisper/models

EXPOSE 5000

CMD ["python", "server.py"]


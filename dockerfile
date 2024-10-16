FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update

RUN apt-get install -y nvidia-container-toolkit

RUN apt-get install -y python3 python3-pip python3-dev

RUN apt-get install -y git git-lfs cmake build-essential libboost-all-dev

RUN apt-get install -y vim

# ---- pytorch (cuda) ----------------------------------
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---- LightGBM (cuda) ----------------------------------
# https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

ENV CMAKE_VERSION=3.30.0

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev

RUN curl -O -L "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(arch).sh" \
    && mkdir /opt/cmake \
    && sh "cmake-${CMAKE_VERSION}-linux-$(arch).sh" --skip-license --prefix=/opt/cmake \
    && ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm "cmake-${CMAKE_VERSION}-linux-$(arch).sh" 

RUN mkdir -p /mnt/workspace && \
    chown $(id -u):$(id -g) /mnt/workspace

WORKDIR /mnt/workspace
RUN git clone --recursive https://github.com/microsoft/LightGBM
RUN apt-get install python3-venv -y
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools numpy scipy scikit-learn

WORKDIR /mnt/workspace/LightGBM
RUN cmake -B build -S . -DUSE_CUDA=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
RUN cmake --build build -j$(nproc)
RUN sh ./build-python.sh install --precompile

# ---- Jupyter lab ----------------------------------
RUN pip install jupyterlab

WORKDIR /app
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

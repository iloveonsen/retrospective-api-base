FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential gcc wget ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev && \
    mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda3 && \
    rm /miniconda.sh

ENV PATH="/miniconda3/bin:${PATH}" \
    CUDA_DOCKER_ARCH=all \
    LAMA_CUBLAS=1

COPY environment.yml .

RUN conda update -n base -c defaults conda -y && \
    conda env create -f environment.yml && \
    conda clean -a -y

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" conda run -n fastapi pip install llama-cpp-python

COPY conda_entrypoint.sh /usr/local/bin/conda_entrypoint.sh
RUN chmod +x /usr/local/bin/conda_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/conda_entrypoint.sh"]

COPY . /app
WORKDIR /app

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Pip deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Core deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/tt-vla

# Copy source code (models/data mounted as volumes)
COPY ttvla/ ttvla/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY run.sh .
RUN chmod +x run.sh

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import peft; print(f'PEFT {peft.__version__}')" && \
    python -m ttvla --help

ENTRYPOINT ["python", "-m", "ttvla"]
CMD ["--help"]

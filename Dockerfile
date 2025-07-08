FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV POETRY_HOME="/opt/poetry" DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        python3.10-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        make \
        cmake \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pipx \
    && pipx install poetry \
    && pipx ensurepath --force
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code
RUN poetry config virtualenvs.in-project false
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install --no-root --no-interaction --no-ansi
COPY . .

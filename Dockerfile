# The docker file is built follwing heavily the instructions here:
# https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
FROM nvidia/cuda:9.2-devel-ubuntu18.04

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Dependency for EGL
RUN apt update && apt install -y cmake build-essential libgl1-mesa-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev

RUN mkdir /workspace/ && cd /workspace/

WORKDIR /workspace

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

WORKDIR /workspace/softgym
# COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
RUN adduser -u 5678 --disabled-password --gecos "" appuser
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["export", "Path=\"/home/jeffrey/miniconda3/bin:$PATH\"", "conda", "activate softgym","python", "examples/random_env.py"]

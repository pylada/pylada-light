#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

# To fully customize the contents of this image, use the following Dockerfile instead:
# https://github.com/microsoft/vscode-dev-containers/tree/v0.117.1/containers/codespaces-linux/.devcontainer/Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/universal:0-linux

# ** [Optional] Uncomment this section to install additional packages. **
#
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update \
   && sudo apt-get -y install --no-install-recommends \
   build-essential gfortran gcc g++ quantum-espresso \
   gfortran quantum-espresso quantum-espresso-data libopenmpi-dev  \
   make git cmake ninja-build \
   #
   # Clean up
   && sudo apt-get autoremove -y \
   && sudo apt-get clean -y \
   && sudo rm -rf /var/lib/apt/lists/*
ARG pyver=3.6
RUN /opt/python/${pyver}/bin/python -m venv /home/codespace/venv
RUN . /home/codespace/venv/bin/activate \
   && pip install --upgrade pip \
   && pip install setuptools wheel scikit-build "cmake!=3.16.3" ninja \
   && pip install cython numpy flake8 mypy black pytest \
   && pip install sphinx sphinx-autobuild rstcheck
ENV DEBIAN_FRONTEND=dialog

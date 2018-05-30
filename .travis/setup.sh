#! /usr/bin/env bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install -q gfortran openmpi-bin openmpi-common libopenmpi-dev gcc-g++ cmake
sudo apt-get install python-scipy python-numpy
sudo pip install --no-cache --upgrade setuptools
sudo pip install --no-cache --upgrade pip
sudo pip install --upgrade numpy
sudo pip install --upgrade --no-cache quantities
sudo pip install --upgrade --no-cache mpi4py
sudo pip install --upgrade --no-cache six
sudo pip install --upgrade --no-cache f90nml
sudo pip install --upgrade --no-cache ipython[all]
sudo pip install --upgrade --no-cache pytest pytest_bdd
sudo pip install --upgrade --no-cache -q Cython
sudo pip install --upgrade --no-cache runipy
sudo pip install --upgrade --no-cache traitlets
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8

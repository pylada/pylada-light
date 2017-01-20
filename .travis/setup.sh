#! /usr/bin/env bash
sudo apt-get install build-essential
sudo apt-get install -q gfortran openmpi-bin openmpi-common libopenmpi-dev
sudo apt-get install python-scipy python-numpy
sudo pip install --no-cache --upgrade setuptools
sudo pip install --no-cache --upgrade pip
sudo pip install --upgrade numpy
sudo pip install --no-cache ipython
sudo pip install --no-cache quantities
sudo pip install --no-cache mpi4py
sudo pip install --no-cache six
sudo pip install --no-cache f90nml
sudo pip install --no-cache ipython
sudo pip install --no-cache pytest pytest_bdd
sudo pip install --no-cache -q Cython
sudo pip install --no-cache runipy
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8

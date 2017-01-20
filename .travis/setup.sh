#! /usr/bin/env bash
sudo apt-get install build-essential
sudo apt-get install -q gfortran openmpi-bin openmpi-common libopenmpi-dev
sudo apt-get install python-scipy python-numpy
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
sudo pip install --upgrade numpy
sudo pip install --no-cache ipython
sudo pip install --no-cache quantities
sudo pip install --no-cache mpi4py
sudo pip install --no-cache six
sudo pip install --no-cache f90nml
sudo pip install --no-cache ipython
sudo pip install pytest pytest_bdd
sudo pip install -q Cython
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -qq
sudo apt-get install -qq g++-4.8

#! /usr/bin/env bash
set -e
set -o

source /etc/bashrc
module load mpi
python3 -m pip install --user -e .
python3 -c "from pylada import test; test()"

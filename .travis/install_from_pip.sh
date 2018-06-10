#! /usr/bin/env bash
set -e
set -o

source ~/.bashrc
module load mpi
python3 -m pip install -e .
python3 -c "from pylada import tests; test()"

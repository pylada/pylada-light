FROM registry.fedoraproject.org/fedora-minimal

VOLUME /project
WORKDIR /project

ARG pyver=3

# quantum-espresso is only needed for the quantum-espresso bindings
# mpi4py is mostly for testing, as well, sometimes, in the more elaborate ways to run jobs on
# clusters.
# ipython stuff is only needed when using the ipython interface
RUN microdnf install -y gcc gcc-c++ gcc-gfortran clang make git \
        quantum-espresso-mpich python$PYVER-devel python$PYVER-mpi4py-mpich

RUN echo "module load mpi" >> /etc/bashrc
RUN python3 -m pip install --user f90nml


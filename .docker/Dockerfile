FROM registry.fedoraproject.org/fedora-minimal

VOLUME /project
WORKDIR /project

ARG pyver=3
ARG compiler=gcc-c++

# quantum-espresso is only needed for the quantum-espresso bindings
# mpi4py is mostly for testing, as well, sometimes, in the more elaborate ways to run jobs on
# clusters.
# ipython stuff is only needed when using the ipython interface
RUN microdnf install -y $compiler gcc-gfortran make git \
        python$pyver-devel python$pyver-mpi4py-mpich

RUN echo "module load mpi" >> /etc/bashrc
RUN python3 -m pip install --user f90nml

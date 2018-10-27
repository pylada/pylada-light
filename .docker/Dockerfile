FROM registry.fedoraproject.org/fedora-minimal

VOLUME /project
WORKDIR /project

# quantum-espresso is only needed for the quantum-espresso bindings
# mpi4py is mostly for testing, as well, sometimes, in the more elaborate ways to run jobs on
# clusters.
# ipython stuff is only needed when using the ipython interface
RUN microdnf install -y gcc gcc-c++ gcc-gfortran clang cmake make git \
        quantum-espresso-mpich \
        python3-numpy python3-devel python3-mpi4py-mpich python3-ipython \
        python3-nbconvert python3-nbformat python3-ipykernel python3-traitlets \
        python3-quantities python3-Cython

RUN echo "module load mpi" >> /etc/bashrc
RUN python3 -m pip install --user f90nml


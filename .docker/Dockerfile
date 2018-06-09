FROM registry.fedoraproject.org/fedora-minimal

VOLUME /project
WORKDIR /project

# quantum-espresso is only needed for the quantum-espresso bindings
# mpi4py is mostly for testing, as well, sometimes, in the more elaborate ways to run jobs on
# clusters.
# ipython stuff is only needed when using the ipython interface
RUN microdnf install -y gcc gcc-c++ gcc-gfortran cmake make clang git \
        quantum-espresso-mpich \
        python2-numpy python2-devel python2-mpi4py-mpich python2-ipython \
        python2-nbconvert python2-nbformat python2-ipykernel python2-traitlets \
        python2-quantities python2-Cython python2-configparser \
        python3-numpy python3-devel python3-mpi4py-mpich python3-ipython \
        python3-nbconvert python3-nbformat python3-ipykernel python3-traitlets \
        python3-quantities python3-Cython

RUN useradd -ms /bin/bash pylada
USER pylada
RUN echo "module load mpi" >> /home/pylada/.bashrc
RUN python2 -m pip install --user f90nml && python3 -m pip install --user f90nml

USER root
RUN microdnf install -y dnf
USER pylada

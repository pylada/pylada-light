
.. _dftcrystal_config_ug: 

Configuring Pylada for CRYSTAL
==============================

Pylada calls CRYSTAL_ as an external program. By default, it expects to find the
'crystal', 'Pcrystal', and 'MPPcrystal' programs in your path. However, it is
possible to configure  this in your ~/.pylada file by modifying the global
parameter :py:data:`~pylada.crystal_program`. For instance, one could add in
~/.pylada:

>>> crystal_program = "/path/to/crystal"

In this case, however, Pylada will not be able to differentiate between serial,
parallel, and massively parallel version of CRYSTAL_. This is affored by
defining :py:data:`~pylada.crystal_program` as a callable. By *default*, it is
the following:

.. code-block: python 

  def crystal_program(self=None, structure=None, comm=None):
    """ Path to serial or mpi or MPP crystal program version. 
    
        If comm is None, then returns the path to the serial CRYSTAL_ program.
        Otherwise, if :py:attr:`dftcrystal.Functional.mpp
        <pylada.dftcrystal.electronic.Electronic.mpp>` is
        True, then returns the path to the MPP version. If that is False, then
        returns the path to the MPI version.
    """
    ser = 'crystal'
    mpi = 'Pcrystal'
    mpp = 'MPPcrystal'
    if self is None or comm is None or comm['n'] == 1: return ser
    if self.mpp is True: return mpp
    return mpi

As can be seen above, Pylada can automatically determine which version of
CRYSTAL_ to call (and does so by default). In the above, ``self`` will be the
functional making the call, or None, ``structure`` will be a crystal structure,
or None, and ``comm`` will be a dictionary indicating how CRYSTAL_ will be run. 

Similarly, a :py:data:`~pylada.properties_program` variable exists which should
contain the path to the properties program.  Finally, there is a
:py:data:`~pylada.crystal_inplace` which controls whether CRYSTAL_ is launched
directly within the output directory, or whithin a temporary directory. The
latter case avoids clutter in the output directories since only a few files are
copied back at the end of a run. If :py:data:`~pylada.crystal_inplace` is False,
then the files are placed in a temporary directory. This temporary directory is
itself located within :envvar:`PYLADA_TMPDIR` (if the environment variable
exists), or within `PBS_TMPDIR` (if that exists), or in the default temporary
directory of the system. A special link `workdir` will be created within the
output for the duration of the crystal run.


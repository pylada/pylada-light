
.. _vasp_config_ug:

Configuring Pylada for VASP
===========================

Pylada calls VASP_ as an external program: all it needs is to know how to call it.
The path to the program can be configured in your ~/.pylada file by simply adding
the following two lines:

>>> vasp_program = "/path/to/vasp" 
>>> is_vasp_4 = True

:py:data:`~pylada.vasp_program` can be an absolute path, or simply the name of the
VASP_ binary if it is available in you ``PATH`` environment variable.
:py:data:`~pylada.is_vasp_4` should be set to True or False depending on which
version of VASP_ is available. It will prevent some vasp-5 only parameters from
being set and will preferentially write the POSCAR_ in a vasp-5 format.

.. note::

   It is also possible to tell the :py:class:`~functional.Vasp` object to use a
   specific version:
   
   >>> vasp = Vasp(program='/path/to/vasp')
   
   It will apply only to calculations launched with that particular instance.
   If vasp does not have a ``program`` attribute, then it uses the global
   definition.

It is possible to run VASP_ in parallel. However, Pylada has to be set up
correctly for running parallel code. Although it needs be done only once, it
can be somewhat painful on some computers or supercomputers. Please see how
to set up :ref:`MPI calculations <configuration_single_mpi_ug>` first.

The last configuration variable is :py:data:`~pylada.verbose_representation`. It
controls whether or not the representation/print-out of the functional should
include parameters which have not changed from the default. It is safest to
keep it True.

.. seealso:: :ref:`pylada-config`

.. _format string: http://docs.python.org/library/string.html#string-formatting

.. _configuration_ug:

Setting up Pylada
*****************

Pylada accepts a range of configuration files:

 1. Files located in the ''config'' sub-directory where Pylada is installed
 2. Files located in one of the directories specified by :envvar:`PYLADA_CONFIG_DIR`
 3. In the user configuration file ''~/.pylada''

Each file is executed and whatever is declared within is placed directly at
the root of the pylada package. The files are read in the order given
above. Within a given directory, files are read alphabetically. Later files
will override previous files, e.g. ''~/.pylada'' will override any
configuration done previously.

Within an  IPython session, or during an actual calculation, the
configuration variables all reside at the root of the :py:mod:`pylada`
package.

..  The following files, located in the ''config'' subdirectory of the source
..  tree, are examples of different configurations: 
.. 
..    - cx1.py: PBSpro + intel mpi
..    - redmesa.py: SLURM + openmpi

Setting up IPython
------------------

IPython_ is Pylada's main interface. In practice Pylada defines an
extension to IPython_. The extension can be automatically loaded when starting IPython by adding 

.. code-block:: python
  
   c.InteractiveShellApp.extensions = [ "pylada" ]

to the ipython_ configuration file (generally,
"$HOME/.config/ipython/profile_default/ipython_config.py").

Another option is to load the extension dynamically for each ipython_ session:

>>> load_ext pylada

.. _configuration_single_mpi_ug:

Running external programs
-------------------------

It is unlikely that users will need to do anything here. However, it is
possible to customize how external programs are launched by Pylada by
redifining :py:meth:`pylada.launch_program`.

.. _configuration_mpi_ug:

Running MPI calculations
------------------------

Pylada can run external MPI software, such as VASP_. Such software must
generally be started through a call to a specific MPI program. It is done
in practice via the configuration variable :py:data:`~pylada.mpirun_exe`.
It can be set as: 

.. code-block:: python

  # openmpi and friends
  mpirun_exe = "mpirun -n {n} {placement} {program}"
  # Crays
  mpirun_exe = "aprun -n {n} {placement} {program}"

:py:data:`~pylada.mpirun_exe` is a `format string`_. It can take any number
of arguments. However, two are required: "n", which is the number of
processes, and "program", which is the commandline for the program to
launch. The latter will be manufactured by Pylada internally. It is a
placeholder at this point. The other reseverved keyword is "ppn", the
number of processes per node. It should only be used for that purpose.
"placement" is useful when running MPI codes side-by-side. Please see below
for extra setup steps required in that case.

The keywords in :py:data:`pylada.mpirun_exe` should be defined in
:py:data:`pylada.default_comm`. This is a dictionary which holds default
values for the different keywords. The dictionary may hold more keywords
than are present in :py:data:`pylada.mpirun_exe`. The converse is not true. It could be for instance:

.. code-block:: python

  default_comm = {'n': 2, 'ppn': 4, 'placement': ''}

Running different MPI calculations side-by-side
-----------------------------------------------

.. currentmodule:: pylada

It is often possible to run calculations side-by-side. One can request 64
processors from a supercomputer and run two VASP_ calculations
simultaneously in the same PBS job. There are a fair number of steps to get this part of Pylada running: 

  1. Set up Pylada to run a single MPI programming as described above
  2. Set the environment variable :py:data:`do_multiple_mpi_programs` to
     True.
  3. Set up :py:data:`figure_out_machines`. This is a string which contains
     a small python script. Pylada runs this python script at the start of
     a PBS/Slurm job to figure out the hostnames of each machine allocated
     to the job. For each *core*, the script should print out a line
     starting with "PYLADA MACHINE HOSTNAME". It will be launched as an MPI
     program on all available cores. By default, it is the following simple
     program:

     .. code-block:: python

         from socket import gethostname
         from mpi import gather, world
         hostname = gethostname()
         results = gather(world, hostname, 0)
         if world.rank == 0:
           for hostname in results:
             print "PYLADA MACHINE HOSTNAME:", hostname
         world.barrier()

     .. note::
       
        This is one of two places where boost.mpi is used. By replacing
        this function with, say, call mpi4py methods, one could remove the
        boost.mpi in Pylada for most use cases. 

     It is important that this function prints out to the standard output
     one line per core (not one line per machine).

     The script is launched by
     :py:meth:`pylada.process.mpi.create_global_comm`.
  4. The names of the machines determined in the previous step are stored
     in :py:data:`default_comm`'s
     :py:attr:`pylada.process.mpi.Communicator.machines` attribute. This is
     simply a dictionary mapping the hostnames determined previously to the
     number of cores. It is possible, however, to modify
     :py:data:`default_comm` after the :py:data:`figure_out_machines`
     script is launched and the results parsed. This is done via the method
     :py:meth:`modify_global_comm`. This method takes a
     :py:class:`pylada.process.mpi.Communicator` instance on input and
     modifes it in-place. By default, this method does nothing.

     On a cray, one could set it up as follows:

     .. code-block:: python

        def modify_global_comm(comm):
          """ Modifies global communicator to work on cray.
      
              Replaces hostnames with the host number. 
          """ 
          for key, value in comm.machines.items():
            del comm.machines[key]
            comm.machines[str(int(key[3:]))] = value

     This would replace the hostnames with something aprun can use for MPI
     placement. :py:meth:`modify_global_comm` is runned once at the
     beginning of a Pylada PBS/Slurm script.

  5. To test that the hostnames where determined correctly, one should copy
     the file "process/tests/globalcomm.py" somewhere, edit it, and launch
     it. The names of the machines should be printed out correctly, with
     the right number of cores:

     .. code-block:: bash

       > cd testhere
       > cp /path/to/pylada/source/process/tests/globalcomm.py
       > vi globalcomm.py
       # This is a PBS script.
       # Modify it so it can be launched.
       > qsub globalcomm.py
       # Then, when it finishes:
       > cat global_comm_out
       EXPECTED N=64 PPN=32
       FOUND
       n 64
       ppn 32
       placement ""
       MACHINES
       PYLADA MACHINE HOSTNAME hector.006 32
       PYLADA MACHINE HOSTNAME hector.006 32
       ...


     The above is an example output. One should try and launch this routine
     on more than one node, with varying number of processes per node, and
     so forth.
 

   6. At this point, Pylada knows the name of each machine participating in
      a PBS/Slurm job. It still needs to be told how to run an MPI job on a
      *subset* of these machines. This will depend on the actual MPI
      implementation installed on the machine. Please first read the manual
      for your machine's MPI implementation.

      Pylada takes care of MPI placements by formatting the
      :py:data:`mpirun_exe` string adequately. For this reason, it is
      expected that :py:data:`mpirun_exe` contains a "{placement}" tag
      which will be replaced with the correct value at runtime. 

      At runtime, before placing the call to an external MPI program, the
      method :py:meth:`pylada.machine_dependent_call_modifier` is called.
      It takes three arguments: a dictionary with which to format the
      :py:data:`mpirun_exe` string, a dictionary or
      :py:data:`pylada.process.mpi.Communicator` instance containing
      information relating to MPI, a dictionary containing the environment
      variables in which to run the MPI program. The first and second
      dictionary will be merged and used to format the
      :py:data:`mpirun_exe` string. By default, this method creates a
      nodefile with only those machines involved in the current job. It
      then sets "placement" to "-machinefile filename" where filename is
      the nodefile. 

      On Crays, one could use the following:

      .. code-block:: python 

         def machine_dependent_call_modifier( formatter=None, 
                                              comm=None, 
                                              env=None ):
           """ Placement modifications for aprun MPI processes. 
              
               aprun expects the machines (not cores) to be given on the
               commandline as a list of "-Ln" with n the machine number.
               """
           from pylada import default_comm
           if formatter is None: return
           if len(getattr(comm, 'machines', [])) == 0: placement = ""
           elif sum(comm.machines.itervalues()) == sum(default_comm.machines.itervalues()):
             placement = ""
           else:
             l = [m for m, v in comm.machines.iteritems() if v > 0]
             placement = "-L{0}".format(','.join(l))
           formatter['placement'] = placement

     Note that the above requires the :py:meth:`pylada.modify_global_comm`
     from point 4.
     
     .. warning::

        All external program calls are routed through this function,
        whether or not it is an MPI program. Hence it is necessary to check
        that the program is to be launched an MPI or not. In the case of
        serial programs, "comm" may be None.

  6. The whole deal can be tested using "process/tests/placement.py"
     This is a PBS job which performs MPI placement on a fake job.
     It should be copied somewhere, edited, and launched. 

     At least two arguments should be set prior to running this script.
     Check the bottom of the script. "ppn" specifies the number of
     processors per nodes. The job should be launched with 2*"ppn" cores.
     "path" should point to the source directory of Pylada. This is so that
     a small program can be found (pifunc) and used for testing. The
     program can be compiled by Pylada by setting "compile_test True" in
     cmake_.

     "placement.py" will launch several *simultaneous* instances of the
     "pifunc" program: one long on three quarters of allocated cores, and
     two smaller calculations on one eigth of the cores each. 

     One should check the ouput to make sure that the programs are running
     side-by-side (not all piled up on the same node), that they are
     runnning simultaneously, and that they run successfully (e.g. mpirun
     does launch them).

.. _configuration_pbs_ug:

Setting up Pylada with PBS/Slurm 
--------------------------------

A ressource manager, such as pbs or slurm, takes care of allocating
supercomputing ressources for submitted jobs. Pylada interfaces with these `via`
a few global data variables:

  - :py:data:`qsub_exe`
  - :py:data:`pbs_string`
  - :py:data:`default_pbs`
  - :py:data:`queues`
  - :py:data:`accounts`
  - :py:data:`debug_queue`

:py:data:`pylada.qsub` defines the executable to submit jobs.

.. code-block:: python

  # openpbs
  qsub_exe = "qsub"
  # slurn
  qsub_exe = "sbatch"

The scripts themselves are defined `via` the `format string`_
:py:data:`pylada.pbs_string`::

  >>> pbs_string =  "#! /bin/bash/\n"\
  ...               "#SBATCH --account={account}\n"\
  ...               "#SBATCH --time={walltime}\n"\
  ...               "#SBATCH -N={nnodes}\n"\
  ...               "#SBATCH -e={err}\n"\
  ...               "#SBATCH -o={out}\n"\
  ...               "#SBATCH -J={name}\n"\
  ...               "#SBATCH -D={directory}\n\n"\
  ...               "python {scriptcommand}\n"

Again, there are few reserved keywords which Pylada will use to fill in the
string.

   - account: defines the relevant account to which the calculation is
     submitted.
   - queue: defines the relevant queue to which the calculation is
     submitted.
   - walltime: maximum time for which the calculation will run
   - n: Number of processes to run the job on.
   - ppn: Number of processes per node.
   - nnodes: number of nodes the calculation will run. This is generally
     computed from "ppn" and "n".
   - err: Standard error file. Generated by Pylada.
   - out: Standard output file. Generated by Pylada.
   - name: Name of the job. Generated by Pylada.
   - directory: Directory where the job is launched. Generated by Pylada. 
   - scriptcommand: Script to launch. Generated by Pylada.

Most of the keywords are automatically generated by Pylada. Is is for the
user to provide a script where the requisite number of keywords make sense
for any particular ressource manager.  

Default keyword values should be stored in the dictionary
:py:data:`pylada.default_pbs`.

The different queues (accounts) accessible to the users can be listed in
:py:data:`pylada.queues` (:py:data:`pylada.accounts`). These will be made
available to the users `via` :ref:`%launch <ipython_launch_ug>`. If "queue"
is not relevant to a particular supercomputer, :py:data:`pylada.queues` can
be set to the empty tuple.

The debug/interactive queue can be made more easily accessible `via`
:py:data:`pylada.debug_queue`::

  pylada.debug_queue = 'queue', 'debug'

The first item of the tuple is the keyword that should be set to access the
relevant resource. The second is the relevant value. These will differ from
supercomputer to supercomputer. In practice, the first is generally "queue"
or "account", and the second is something like "debug".

It is also possible to define :py:data:`pbs_string` as a callable which
takes keyword arguments and return a string.
For instance, PBSpro does not accept names longuer that fifteen characters.
That's just to high-tech for an expensive propietary software::


  default_pbs = { 'walltime': "00:55:00", 'nnodes': 1, 'ppn': 32,
                  'account': 'eO5', 'header': "", 'footer': "" }
  
  def pbs_string(**kwargs):
    if 'name' in kwargs:
      kwargs['name'] = kwargs['name'][:min(len(kwargs['name']), 15)]
    return "#! /bin/bash --login\n"                                      \
           "#PBS -e \"{err}\"\n"                                         \
           "#PBS -o \"{out}\"\n"                                         \
           "#PBS -N {name}\n"                                            \
           "#PBS -l mppwidth={n}\n"                                      \
           "#PBS -l mppnppn={ppn}\n"                                     \
           "#PBS -l walltime={walltime}\n"                               \
           "#PBS -A {account}\n"                                         \
           "#PBS -V \n\n"                                                \
           "export PYLADA_TMPDIR=/work/e05/e05/`whoami`/pylada_tmp\n"    \
           "if [ ! -e $PYLADA_TMPDIR ] ; then\n"                         \
           "  mkdir -p $PYLADA_TMPDIR\n"                                 \
           "fi\n"                                                        \
           "cd {directory}\n"                                            \
           "{header}\n"                                                  \
           "python {scriptcommand}\n"                                    \
           "{footer}\n".format(**kwargs)

.. _install_vasp_ug:

Setting up Pylada to call VASP
==============================

There are only two variable specific to vasp calculations:
   
 - :py:data:`~pylada.is_vasp_4` defines whether the installed vasp program is
   version 4.6 or 5.0 and higher. In practice, this determines which POSCAR
   format to use, and whether or not some input options are available.
 - :py:data:`~pylada.vasp_program` defines the vasp executable. In general, it
   will be a string with path to the executable. It can also be a callable
   which takes the functional as input:

   .. code-block:: python

      def vasp_program(self):
        """ Figures out the vasp executable. 
        
            It is expected that two vasp executable exist, a *normal* vasp,
            and a one compiled for non-collinear calculations.
        """
        lsorbit = getattr(self, 'lsorbit', False) == True
        return "vasp-4.6-nc" if lsorbit  else "vasp-4.6"

.. note::

   Pylada should be :ref:`Set up <configuration_mpi_ug>` properly to run mpi calculations.

.. warning::

   Please follow the links for their description. Questions regarding how to
   compile should be addressed to the relevant authorities.

.. _format string: http://docs.python.org/library/st dtypes.html#str.format

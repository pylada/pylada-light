.. _configuration_ug:

Setting up Pylada
*****************

Pylada accepts a range of python configuration files:

 1. Files located in the ''config'' sub-directory where Pylada is installed
 2. Files located in one of the directories specified by :envvar:`PYLADA_CONFIG_DIR`
 3. In the user configuration file ''~/.pylada''

*Every* python file in these locations, regardless of its name,  is executed and *whatever* is declared within is placed directly at
the root of the pylada package.  For example if you define ``my_var = 7`` in ''~/.pylada'', then when you import
pylada, ``pylada.my_var`` will be defined and equal 7.
The files are read in the order given
above. Within a given directory, files are read alphabetically. Later files
will override previous files, e.g. ''~/.pylada'' will override any
configuration done previously.

Within an  IPython session, or during an actual calculation, the
configuration variables all reside at the root of the :py:mod:`pylada`
package.

For example, the following files, located in the ''config'' subdirectory of the source
tree, are examples of different configurations: 
 
    - cx1.py: PBSpro + intel mpi
    - redmesa.py: SLURM + openmpi

These files define variables, for example, such as 'pbs_string', (a template string to be used in constructing 
submissions to the PBS batch queueing system (see below, :ref:`configuration_pbs_ug`)) that are then available in the
pylada global namespace (e.g., just :py:meth:`pylada.pbs_string`, not :py:meth:`pylada.config.pbs_string` or anything else).

.. note::

   The variables defined in these configurations files are critical for the proper functioning of pylada, especially its 
   ability to interact with your batch queueing system.  There are defaults for all these variables, but if your jobs are
   not running, it is likely that one of these variables/functions needs to be specifically customized for your system.  See below 
   (:ref:`Setting up Pylada with PBS/Slurm <configuration_pbs_ug>`) for
   details of the relevant variables.

   **A strategy for debugging your pylada setup** 
   Let pylada generate the input files for you via, e.g. :py:meth:`launch scattered`.  When your run 
   fails, you will be left with a script that was submitted to your queueing system.  This script contains commands to run your job.  From
   an interactive node, run and edit the commands in this script directly until they work.  Now submit the script.  Submit and edit the
   script until it works. *Finally*, edit the pylada variables described here until they generate the script that works.  Pylada's default
   parameters provie a *template* that you probably will need to adjust to suit your particular system.

Setting up IPython
------------------

IPython_ is Pylada's main interface. In practice Pylada defines an
extension to IPython_. The extension can be automatically loaded when starting IPython by adding 

.. code-block:: python
  
   c.InteractiveShellApp.extensions = [ "pylada.ipython" ]

to the ipython_ configuration file (generally,
"$HOME/.config/ipython/profile_default/ipython_config.py").

Another option is to load the extension dynamically for each ipython_ session:

>>> load_ext pylada.ipython

Please see :ref:`IPython high-throughput interface <ipython_ug>` for usage of the IPython_ interface.

.. _configuration_single_mpi_ug:

Running external programs
-------------------------

It is unlikely that users will need to do anything here. However, it is
possible to customize how external programs are launched by Pylada by
redifining :py:meth:`pylada.launch_program`.  The default function launches external
programs using python's ``Popen`` function

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
placeholder at this point. The other reserved keyword is "ppn", the
number of processes per node. It should only be used for that purpose.
"placement" is useful when running MPI codes side-by-side. Please see below
for extra setup steps required in that case.

The keywords in :py:data:`pylada.mpirun_exe` should be defined in
:py:data:`pylada.default_comm`. This is a dictionary which holds default
values for the different keywords. The dictionary may hold more keywords
than are present in :py:data:`pylada.mpirun_exe`. The converse is not true (all keys in :py:data:`pylada.mpirun_exe`
*must* be defined). It could be, for instance:

.. code-block:: python

  default_comm = {'n': 2, 'ppn': 4, 'placement': ''}

For instructions on the advanced feature of running multiple mpi jobs side by side, please see
:ref:`Running different MPI calculations side-by-side <side_by_side_mpi_ug>`

.. _configuration_pbs_ug:

Setting up Pylada with PBS/Slurm 
--------------------------------

A resource manager, such as pbs or slurm, takes care of allocating
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

.. note::

  It has been observed on some systems (openmpi on CentOS) that the ppn flag is critical, for example,
  :py:data:`pylada.pbs_string` needs to include the line

  #PBS -l nodes={nnodes}:ppn={ppn}

  Discovering this problem involved understanding the role of :py:data:`figure_out_machines`, described briefly
  in :ref:`Running different MPI calculations side-by-side <side_by_side_mpi_ug>`.   

Most of the keywords are automatically generated by Pylada. It is for the
user to provide a script where the requisite number of keywords make sense
for any particular resource manager.  

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

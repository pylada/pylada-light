###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
# 
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
# 
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
# 
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
# 
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

mpirun_exe = "mpirun -n {n} {placement} {program}"
""" Command-line to launch external mpi programs. """
def machine_dependent_call_modifier(formatter=None, comm=None, env=None):
  """ Machine dependent modifications. 
  
      This is a fairly catch all place to put machine dependent stuff for mpi
      calls, including mpi placement.

      The formatter used to format the :py:data:`~pylada.mpirun_exe` string is
      passed as the first argument. It can be modified *in-place* for machine
      dependent stuff, or for mpi placement. The latter case occurs only if
      ``comm`` has a non-empty ``machines`` attribute. In that case,
      :py:attr:`~pylada.process.mpi.machines` is a dictionary mapping the
      hostnames to the number of procs on that host. Finally, an dictionary
      containing the environment variables can also be passed. It should be
      modified *in-place*.

      By default, the 'placement' value of the formatter is modified to reflect
      the nodefile of a specific mpi placement. This occurs only if
      mpi-placement is requested (eg `comm.machines` exists and is not empty).

      This function is called only from :py:function:`pylada.launch_program`. If
      calls fail, it is a good idea to copy :py:function:`pylada.launch_program`
      into your $HOME/.pylada and debug it from there.

      :param dict formatter:
        Dictionary used in formatting the command line of
        :py:function:`~pylada.launch`. It should be modified *in-place*.
      :param comm:
        Communicator used in this particular calculation. At this point in
        :py:function:`~pylada.launch_program`, dictionary data from the
        communicator have been copied to the formatter. It is passed here in
        case its attributes :py:attr:`~pylada.process.mpi.Communicator.machines`
        or the nodefile returned by
        :py:method:`~pylada.process.mpi.Communicator.nodefile`
        is needed. However, the communicator itself should not be modified.
      :type comm: :py:class:`~pylada.process.mpi.Communicator`
      :param dict env: 
        Dictionary of environment variables in which to run the call.

      :return: ignored
  """
  from pylada.misc import bugLev
  if len(getattr(comm, 'machines', [])) != 0:
    nfile = comm.nodefile()
    formatter['placement'] = "-machinefile {0}".format( nfile)
    if bugLev >= 5:
      print "config/mpi: machine_dep_call_mod: nodefile: \"%s\"" % (nfile,)
      with open( nfile) as fin:
        print "config/mpi: machine_dep_call_mod: nodefile contents: \"%s\"" \
          %(fin.read(),)

def modify_global_comm(communicator):
  """ Modifies global communicator so placement can be done correctly. 
  
      This function is called by :py:func:`create_global_comm`. It can be used
      to modify the global communicator to work better with a custom placement
      function.
  """
  pass

def launch_program( cmdl, comm=None, formatter=None, env=None, 
                    stdout=None, stderr=None, stdin=None, outdir=None ):
  """ Command used to launch a program.
  
      This function launches external programs for Pylada. It is included as a
      global so that it can be adapted to different computing environment. It
      also makes it easier to debug Pylada's mpi configuration when installing on
      a new machine.

      .. note::

        The number one configuration problem is an incorrect
        :py:data:`~pylada.mpirun_exe`.

      .. note::
       
        The number two configuration problem is mpi-placement (eg how to launch
        two different mpi program simultaneously in one PBS/SLURM job). First
        read the manual for the mpi environment on the particular machine Pylada
        is installed on. Then adapt
        :py:function:`~pylada.machine_dependent_call_modifier` by redeclaring it
        in $HOME/.pylada.

      :param str cmld: 
        Command-line string. It will be formatted using ``formatter`` or
        ``comm`` if either are present. Otherwise, it should be exactly the
        (bash) command-prompt.
      :param comm: 
        Should contain everythin needed to launch an mpi call. 
        In practice, it is copied from :py:data:`~pylada.default_comm` and
        modified for the purpose of a particular call (e.g. could use fewer
        than all available procs)
      :type comm: :py:class:`~pylada.process.mpi.Communicator`
      :param dict formatter:
        Dictionary with which to format the communicator. If ``comm`` is
        present, then it will be updated with ``comm``'s input.
      :param dict env: 
        Dictionary containing the environment variables in which to do call.
      :param stdout:
        File object to which to hook-up the standard output. See Popen_.
      :param stderr:
        File object to which to hook-up the standard error. See Popen_.
      :param str outdir:
        Path to the working directory.

      .. _Popen:: http://docs.python.org/library/subprocess.html#popen-constructor
  """
  from shlex import split as shlex_split
  from subprocess import Popen
  from pylada import machine_dependent_call_modifier
  from pylada.misc import Changedir
  from pylada.misc import bugLev
  from pylada.misc import testValidProgram

  if bugLev >= 5:
    print "config/mpi: launch_program: entry"
    print "  parm cmdl: %s  type: %s" % (cmdl, type(cmdl),)
    print "  parm comm: %s  type: %s" % (comm, type(comm),)
    print "  parm formatter: %s  type: %s" % (formatter, type(formatter),)
    print "  parm env: %s  type: %s" % (env, type(env),)
    print "  parm outdir: %s  type: %s" % (outdir, type(outdir),)

  # At this point formatter is {"program": vasp}
  # and cmdl is "mpirun -n {n} {placement} {program}"

  # Set in formatter: 'placement': '', 'ppn': 8, 'n': 8
  # make sure that the formatter contains stuff from the communicator, eg the
  # number of processes.
  if comm is not None and formatter is not None:
    formatter.update(comm)
  if bugLev >= 5:
    print "config/mpi: launch_program: comm mod formatter: %s" % (formatter,)

  # Set in formatter: 'placement': '-machinefile /home.../pylada_commtempfile'
  # Stuff that will depend on the supercomputer.
  machine_dependent_call_modifier(formatter, comm, env)
  if bugLev >= 5:
    print "config/mpi: launch_program: mach mod formatter: %s" % (formatter,)

  if bugLev >= 5:
    print "config/mpi: launch_program: plac formatter: %s" % (formatter,)

  # if a formatter exists, then use it on the cmdl string.
  if formatter is not None: cmdl = cmdl.format(**formatter)
  # otherwise, if comm is not None, use that.
  elif comm is not None: cmdl = cmdl.format(**comm)

  # Split command from string to list
  if bugLev >= 1:
    print "config/mpi: launch_program: final full cmdl: \"%s\"" % (cmdl,)
  cmdl = shlex_split(cmdl)
  if bugLev >= 1:
    print "config/mpi: launch_program: final split cmdl: %s" % (cmdl,)
    print "config/mpi: launch_program: final stdout: %s" % (stdout,)
    print "config/mpi: launch_program: final stderr: %s" % (stderr,)
    print "config/mpi: launch_program: final stdin: %s" % (stdin,)
    print "config/mpi: launch_program: final outdir: \"%s\"" % (outdir,)
    print "config/mpi: launch_program: final env: %s" % (env,)

  # makes sure the directory exists:
  if outdir is not None:
    with Changedir(outdir) as cwd: pass

  # Finally, start the process.
  popen = Popen( cmdl, stdout=stdout, stderr=stderr, stdin=stdin,
    cwd=outdir, env=env )
  if bugLev >= 1:
    print "config/mpi: launch_program: popen: %s" % (popen,)
    print "config/mpi: launch_program: popen.pid: %s" % (popen.pid,)
  if testValidProgram: popen.wait()
  return popen



default_comm = {'n': 2, 'ppn': 4, 'placement': ''}
""" Default communication dictionary. 

    should contain all key-value pairs used in :py:data:`mpirun_exe`.  In a
    script which manages mpi processes, it is also the global communicator. In
    other words, it is the one which at the start of the application is given
    knowledge of the machines (via :py:func:`~pylada.create_global_comm`). Other
    communicators will have to acquire machines from this one. In that case, it
    is likely that 'n' is modified.
"""

# pbs/slurm related stuff.
queues = ()
""" List of slurm or pbs queues allowed for use. 

    This is used by ipython's %launch magic function. 
    It is not required for slurm systems. 
    If empty, then %launch will not have a queue option.
"""
###accounts = ['CSC000', 'BES000']
accounts = ['']
""" List of slurm or pbs accounts allowed for use. 

    This is used by ipython's %launch magic function. 
    It is not required for slurm systems. 
    If empty, then %launch will not have a queue option.
"""

debug_queue = "queue", "debug"
""" How to select the debug queue. 

    First part of the tuple is the keyword argument to modify when calling
    the pbs job, and the second is its value.
"""


qsub_exe = "qsub"
""" Qsub/sbatch executable. """
qsub_array_exe = None
""" Qsub for job arrays.

    If not None, if should be a tuple consisting of the command to launch job
    arrays and the name of the environment variable holding the job index. 

    >>> qsub_array_exe = 'qsub -J 1-{nbjobs}', '$PBS_ARRAY_INDEX'

    The format ``{array}`` will receive the arrays to launch.
"""


# qdel_exe = 'scancel'
qdel_exe = 'mjobctl -c'
""" Qdel/scancel executable. """

default_pbs = {
  ###'account': accounts[0],
  'walltime': "00:30:00",
  'nnodes': 1,
  'ppn': 1,
  'header': '',
  'footer': ''
}
""" Defaults parameters filling the pbs script. """

#pbs_string =  '''#!/bin/bash
##SBATCH --account={account}
##SBATCH --time={walltime}
##SBATCH -N {nnodes}
##SBATCH -e {err}
##SBATCH -o {out}
##SBATCH -J {name}
##SBATCH -D {directory}
#
#echo config/mpi.py pbs_string: header: {header}
#echo config/mpi.py pbs_string: scriptcommand: python {scriptcommand}
#echo config/mpi.py pbs_string: footer: {footer}
#
#{header}
#python {scriptcommand}
#{footer}
#
#'''


pbs_string =  '''#!/bin/bash
#PBS -A {account}
#PBS -q {queue}
#PBS -m n
#PBS -l walltime={walltime}
#PBS -l nodes={nnodes}
#PBS -l feature=24core
#PBS -e {err}
#PBS -o {out}
#PBS -N {name}
#PBS -d {directory}

cd {directory}

echo config/mpi.py pbs_string: header: {header}
echo config/mpi.py pbs_string: scriptcommand: python {scriptcommand}
echo config/mpi.py pbs_string: footer: {footer}

echo config/mpi.py pbs_string: directory: {directory}
echo config/mpi.py pbs_string: which python A: $(which python)


module load epel/6.3
module load python/2.7.4/impi-intel

. /nopt/nrel/ecom/cid/pylada/5.0.006/virtipy/bin/activate


export PYTHONPATH=$PYTHONPATH:/nopt/nrel/ecom/cid/pylada/5.0.006/pinstall/lib64/python2.7/site-packages


echo ''
echo config/mpi.py pbs_string: which python B: $(which python)

echo ''
echo config/mpi.py pbs_string: module list:
module list 2>&1

echo ''
echo config/mpi.py: PATH: $PATH

echo ''
echo config/mpi.py: PYTHONPATH: $PYTHONPATH

echo ''
echo config/mpi.py === begin printenv
printenv
echo config/mpi.py === end printenv

echo ''
echo config/mpi.py === begin sorted printenv
printenv | sort
echo config/mpi.py === end sorted printenv

echo config/mpi.py === begin cat nodefile
cat $PBS_NODEFILE
echo config/mpi.py === end cat nodefile

python -c 'import argparse'
echo config/mpi.py pbs_string: after test argparse

python -c 'import numpy'
echo config/mpi.py pbs_string: after test numpy

python -c 'import quantities'
echo config/mpi.py pbs_string: after test quantities

python -c 'import mpi4py'
echo config/mpi.py pbs_string: after test mpi4py


{header}
python {scriptcommand}
{footer}


'''
""" Default pbs/slurm script. """






do_multiple_mpi_programs = True
""" Whether to get address of host machines at start of calculation. """

# Figure out machine hostnames for a particular job.
# Can be any programs which outputs each hostname (once per processor),
# preceded by the string "PYLADA MACHINE HOSTNAME:"

figure_out_machines =  '''
from socket import gethostname
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


hostname = gethostname()
names = comm.gather( hostname, root=0)
if rank == 0:
  for nm in names:
    print "PYLADA MACHINE HOSTNAME:", nm

#if bugLev >= 5:
#  fname = os.getenv("HOME") + "/temp.figure_out_machines.%03d" % (rank,)
#  fdebug = open( fname, "w")
#  print >> fdebug, \
#    "config/mpi.py: figure_out_machines: size: %d  rank: %d  hostname: %s" \
#    % (size, rank, hostname,)
#
#  if rank == 0:
#    for nm in names:
#      print >> fdebug, "config/mpi.py: figure_out_machines: nm: %s" % (nm,)
#  fdebug.close()

'''

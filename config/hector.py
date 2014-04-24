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

debug_queue = None
""" No debug queue on cx1. """

qsub_exe = "qsub"
""" Qsub executable. """
qsub_array_exe = "qsub -r y -J 1-{nbjobs}", "$PBS_ARRAY_INDEX"
""" Launches job-arrays. """
qdel_exe = "qdel"
""" Qdel executable. """
          
default_pbs = { 'walltime': "00:55:00", 'nnodes': 1, 'ppn': 32,
                'account': 'eO5', 'header': "", 'footer': '' }
""" Defaults parameters filling the pbs script. """

def pbs_string(**kwargs):
  """ Returns pbs script. """
  if 'name' in kwargs:
    kwargs['name'] = kwargs['name'][:min(len(kwargs['name']), 15)]
  return "#! /bin/bash --login\n"                                              \
         "#PBS -e \"{err}\"\n"                                                 \
         "#PBS -o \"{out}\"\n"                                                 \
         "#PBS -N {name}\n"                                                    \
         "#PBS -l mppwidth={n}\n"                                              \
         "#PBS -l mppnppn={ppn}\n"                                             \
         "#PBS -l walltime={walltime}\n"                                       \
         "#PBS -A {account}\n"                                                 \
         "#PBS -V \n\n"                                                        \
         "export PYLADA_TMPDIR=/work/e05/e05/`whoami`/pylada_tmp\n"            \
         "if [ ! -e $PYLADA_TMPDIR ] ; then\n"                                 \
         "  mkdir -p $PYLADA_TMPDIR\n"                                         \
         "fi\n"                                                                \
         "cd {directory}\n"                                                    \
         "{header}\n"                                                          \
         "python {scriptcommand}\n"                                            \
         "{footer}\n".format(**kwargs)

default_comm = { 'n': 2, 'ppn': default_pbs['ppn']}
""" Default mpirun parameters. """

mpirun_exe = "aprun -n {n} {placement} {program} "
""" Command-line to launch external mpi programs. """
do_multiple_mpi_program = True
""" Whether setup to lauch multiple MPI programs. """
accounts = ['e05-power-nic', 'e05-qmdev-nic']

def machine_dependent_call_modifier(formatter=None, comm=None, env=None):
  """ Placement modifications for aprun MPI processes. 
     
      aprun expects the machines (not cores) to be given on the commandline as
      a list of "-Ln" with n the machine number.
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

def modify_global_comm(comm):
  """ Modifies global communicator to work on cray.

      Replaces hostnames with the host number. 
  """ 
  for key, value in comm.machines.items():
    del comm.machines[key]
    comm.machines[str(int(key[3:]))] = value

def vasp_program(self):
  """ Signifies the vasp executable. 
  
      It is expected that two vasp executable exist, a *normal* vasp, and a one
      compiled for non-collinear calculations.

  """
  lsorbit = getattr(self, 'lsorbit', False) == True
  return "vasp-4.6-nc" if lsorbit  else "vasp-4.6"

def ipython_qstat(self, arg):
  """ Prints jobs of current user. """
  from subprocess import Popen, PIPE
  from IPython.utils.text import SList
  from itertools import chain
  # get user jobs ids
  whoami = Popen(['whoami'], stdout=PIPE).communicate()[0].rstrip().lstrip()
  jobs   = Popen(['qstat', '-u', whoami], stdout=PIPE).communicate()[0].split('\n')
  if len(jobs) == 1: return
  ids    = SList(jobs[5:-1]).fields(0)
  # now gets full info
  jobs   = SList(Popen(['qstat', '-f'] + ids, stdout=PIPE).communicate()[0].split('\n'))
  names  = [u[u.find('=')+1:].lstrip().rstrip() for u in jobs.grep('Job_Name')]
  mpps   = [int(u[u.find('=')+1:]) for u in jobs.grep('Resource_List.ncpus')]
  states = [u[u.find('=')+1:].lstrip().rstrip() for u in jobs.grep('job_state')]
  ids    = [u[u.find(':')+1:].lstrip().rstrip() for u in jobs.grep('Job Id')]
  # the result is then  synthesized, with the first field the job id, and the
  # last the job name. This is important since it will be used by cancel as such.
  return SList([ "{0:>10} {1:>4} {2:>3} -- {3}".format(id, mpp, state, name)   \
                 for id, mpp, state, name in zip(ids, mpps, states, names)]) 

def crystal_program(self=None, structure=None, comm=None):
  """ Path to serial or mpi or MPP crystal program version. 
  
      If comm is None, then returns the path to the serial CRYSTAL_ program.
      Otherwise, if :py:attr:`dftcrystal.Functional.mpp
      <pylada.dftcrystal.electronic.Electronic.mpp>` is
      True, then returns the path to the MPP version. If that is False, then
      returns the path to the MPI version.
  """
  ser = getattr(self, 'program_ser', None) 
  mpi = getattr(self, 'program_mpi', None)
  mpp = getattr(self, 'program_mpp', None)
  if ser is None: ser = 'crystal'
  if mpi is None: mpi = 'Pcrystal'
  if mpp is None: mpp = 'MPPcrystal'

  if self is None or comm is None: return ser
  if self.mpp is True: return mpp
  return mpi

crystal_inplace = False

global_tmpdir='$WORK/pylada_tmp'

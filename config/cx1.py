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
          
default_pbs = { 'walltime': "00:55:00", 'nnodes': 1, 'ppn': 4}
""" Defaults parameters filling the pbs script. """

def pbs_string(**kwargs):
  """ Returns pbs script. """
  if 'name' in kwargs:
    kwargs['name'] = kwargs['name'][:min(len(kwargs['name']), 15)]
  return "#! /bin/bash\n"\
             "#PBS -e \"{err}\"\n"\
             "#PBS -o \"{out}\"\n"\
             "#PBS -m n\n"\
             "#PBS -N {name}\n"\
             "#PBS -l select={nnodes}:ncpus={n}\n"\
             "#PBS -l walltime={walltime}\n"\
             "#PBS -V \n\n"\
             "cd {directory}\n"\
             "python {scriptcommand}\n".format(**kwargs)

default_comm = { 'n': 2, 'ppn': default_pbs['ppn']}
""" Default mpirun parameters. """

mpirun_exe = "mpirun -np {n} {program}"
""" Command-line to launch external mpi programs. """
do_multiple_mpi_program = True
""" Whether setup to lauch multiple MPI programs. """

def machine_dependent_call_modifier(formatter=None, comm=None, env=None):
  """ Placement modifications for MPI processes. 
  
      Nodefile is re-written with one hostname per line and per processor
      (rather than  per host, with 'slots=n' arguments indicating the number of
      procs per node). Finally, the environment variable ``PBS_NODEFILE`` is
      modified to point to the new nodefile. 

      .. note:: 
      
         Also, the hostname were shortened to exclude cx1.hpc.imperial.ac.uk
         domain name in :py:function:`~pylada.modify_global_comm`. 
  """
  if len(getattr(comm, 'machines', [])) == 0: return ""

  # modify nodefile
  nodefile = comm.nodefile()
  with open(nodefile, 'w') as file:
    for key, value in comm.machines.iteritems():
      stg = '\n'.join([key]*value) + '\n'
      if bugLev >= 5:
        print "config/cx1: nodefile: key: \"%s\"  value: \"%s\"  stg: \"%s\"" \
          % ( key, value, stg,)
      file.write(stg)
  # modify environment variable
  env['PBS_NODEFILE'] = nodefile

  return "PBS_NODEFILE={0}".format(nodefile)

def modify_global_comm(comm):
  """ Modifies global communicator to work on cx1. 

      Somehow, intel's mpi does not like fully qualified domain-names, eg
      cx1.hp.imperial.ac.uk. Communicator is modified *in-place*.
  """ 
  for key, value in comm.machines.items():
    del comm.machines[key]
    comm.machines[key[:key.find('.')]] = value

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
  # get user jobs ids
  jobs   = SList(Popen(['qstat', '-f'], stdout=PIPE).communicate()[0].split('\n'))
  names  = [u[u.find('=')+1:].lstrip().rstrip() for u in jobs.grep('Job_Name')]
  mpps   = [int(u[u.find('=')+1:]) for u in jobs.grep('Resource_List.ncpus')]
  states = [u[u.find('=')+1:].lstrip().rstrip() for u in jobs.grep('job_state')]
  ids    = [u[u.find(':')+1:].lstrip().rstrip() for u in jobs.grep('Job Id')]
  return SList([ "{0:>10} {1:>4} {2:>3} -- {3}".format(id, mpp, state, name) \
                 for id, mpp, state, name in zip(ids, mpps, states, names)])

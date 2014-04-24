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

if "pyladabase" in globals()["pyladamodules"]:
  local_push_dir = "/projects/nrel/cid/database_tmp"
  """ Local directory where database stuff is pushed. """
if "jobs" in globals()["pyladamodules"]:
  template_pbs = globals()["default_slurm"]
  """ Template pbs script to use. Depends on machine. """
  debug_queue = "queue", "inter"
  """ How to select the debug queue. 

      First part of the tuple is the keyword argument to modify when calling
      the pbs job, and the second is its value.
  """
  accounts = [] 
  """ List of slurm or pbs accounts allowed for use. 

      This is used by ipython's %launch magic function. 
      It is not required for slurm systems. 
      If empty, then %launch will not have a queue option.
  """
  qsub_exe = "sbatch"
  """ Qsub executable. """
  resource_string = "-N {nnodes}"
  """ Format string to specify computational resources. 
      
      The first argument is total number of processes, the second the number of
      nodes itself, the third the number of processes per node.
  """
  mpirun_exe = "mpirun -np {nprocs} numa_wrapper -ppn={ppernode} {program}"
  """ Command-line to launch external mpi programs. """
  cpus_per_node = 8
  """ Number of processes per node. """
  pylada_with_slurm = True
  """ If True use slurm as ressource manager, else use openpbs. """

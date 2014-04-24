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

debug_queue = "queue", "inter"
""" How to select the debug queue. 

    First part of the tuple is the keyword argument to modify when calling
    the pbs job, and the second is its value.
"""

accounts = ["BES000"]
""" List of slurm or pbs accounts allowed for use. 

    This is used by ipython's %launch magic function. 
    It is not required for slurm systems. 
    If empty, then %launch will not have a queue option.
"""

qsub_exe = "sbatch"
""" Qsub executable. """
          
default_pbs = { 'account': accounts[0], 'walltime': "06:00:00", 'nnodes': 1, 'ppn': 8}
""" Defaults parameters filling the pbs script. """

pbs_string =  "#! /bin/bash\n"\
              "#SBATCH --account={account}\n"\
              "#SBATCH --time={walltime}\n"\
              "#SBATCH -N {nnodes}\n"\
              "#SBATCH -e \"{err}.%j\"\n"\
              "#SBATCH -o \"{out}.%j\"\n"\
              "#SBATCH -J {name}\n"\
              "#SBATCH -D {directory}\n\n"\
              "python {scriptcommand}\n"
""" Default slurm script. """

default_comm = { 'n': 2, 'ppn': default_pbs['ppn'] }
""" Default mpirun parameters. """

mpirun_exe = "mpirun -np {n} {placement} numa_wrapper -ppn={ppn} {program}"
""" Command-line to launch external mpi programs. """

def ipython_qstat(self, arg):
  """ squeue --user=`whoami` -o "%7i %.3C %3t  --   %50j" """
  from subprocess import Popen, PIPE
  from IPython.utils.text import SList
  from getpass import getuser

  # finds user name.
  whoami = getuser()
  squeue = Popen(["squeue", "--user=" + whoami, "-o", "\"%7i %.3C %3t    %j\""], stdout=PIPE)
  result = squeue.stdout.read().rstrip().split('\n')
  result = SList([u[1:-1] for u in result[1:]])
  return result.grep(str(arg[1:-1]))

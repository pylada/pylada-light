# example slurm configuration file
vasp_has_nlep = False
qsub_exe="sbatch"
qdel_exe = "scancel"
do_multiple_mpi_programs=False

default_comm = {'n': 2, 'placement': '', 'ppn': 16}
""" Default mpirun parameters. """
mpirun_exe = "mpirun -np {n} {placement} -npernode {ppn} {program}"
""" Command-line to launch external mpi programs. """

def ipython_qstat(self, arg):
  """ squeue --user=`whoami` -o "%7i %.3C %3t  --   %50j" """
  from six import PY3
  from subprocess import Popen, PIPE
  from IPython.utils.text import SList
  from getpass import getuser
  whoami = getuser()
  squeue = Popen(["squeue", "--user=" + whoami, "-o", "\"%7i %.3C %3t    %j\""], stdout=PIPE)
  result = squeue.stdout.read().rstrip().splitlines()
  if PY3:
    result = SList([u[1:-1].decode("utf-8") for u in result[1:]])
  else:
    result = SList([u[1:-1] for u in result[1:]])

  return result if str(arg) == '' else result.grep(str(arg[1:-1]))

pbs_string =  """#!/bin/bash
###SBATCH --account={account}
#SBATCH --time={walltime}
#SBATCH -N {nnodes}
#SBATCH --ntasks-per-node={ppn}
#SBATCH --partition={queue}
#SBATCH -e {err}
#SBATCH -o {out}
#SBATCH -J {name}
#SBATCH -D {directory}

{header}
python {scriptcommand}
{footer}
"""

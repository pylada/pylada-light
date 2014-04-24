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

#!/usr/bin/python
#PBS -e global_comm_err
#PBS -o global_comm_out
#PBS -N global_comm
#PBS -l mppwidth=64
#PBS -l walltime=00:02:00
#PBS -A e05-qmdev-nic
#PBS -V 

def test(nbprocs, ppn):
  import pylada
  from pylada.process.mpi import create_global_comm

  pylada.default_comm['ppn'] = ppn
  pylada.default_comm['n'] = nbprocs
  print 'EXPECTED N={0}, PPN={1}'.format(nbprocs, ppn)
  create_global_comm(nbprocs)
  print 'FOUND'
  for u in pylada.default_comm.iteritems(): print u[0], u[1]
  print 'MACHINES'
  for u in pylada.default_comm.machines.iteritems(): print u[0], u[1]

if __name__ == '__main__': 
  test(nbprocs=64, ppn=32)

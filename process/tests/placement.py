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

#! /work/y07/y07/nag/python-shared-xe6/2.6.5/bin/python
#PBS -e /home/e05/e05/mdavezac/bull/placement_err
#PBS -o /home/e05/e05/mdavezac/bull/placement_out
#PBS -N placement_comm
#PBS -l mppwidth=64
#PBS -l walltime=00:02:00
#PBS -A e05-qmdev-nic
#PBS -V

def test(nbprocs, ppn, executable):
  from os import getcwd
  print 'IN DIRECTORY', getcwd()
  from pylada.process.mpi import create_global_comm
  from pylada.process.iterator import IteratorProcess
  from functional import Functional
  import pylada

  print 'CREATING GLOBAL COMM'
  pylada.default_comm['ppn'] = ppn
  pylada.default_comm['n'] = nbprocs
  create_global_comm(nbprocs)

  print 'CREATING FUNCTIONALS AND PROCESSES'
  lfunc = Functional(executable, range(ppn*10, ppn*10+8))
  long = IteratorProcess(lfunc, outdir='long')

  sfunc = Functional(executable, [10])
  short0 = IteratorProcess(sfunc, outdir='short0')
  short1 = IteratorProcess(sfunc, outdir='short1') 
  
  print 'CREATING COMMUNICATORS'
  long_comm = pylada.default_comm.lend(3*(nbprocs//4))
  assert len(long_comm.machines) == 2
  short_comm0 = pylada.default_comm.lend(pylada.default_comm['n']//2)
  assert len(short_comm0.machines) == 1
  short_comm1 = pylada.default_comm.lend('all')
  assert len(short_comm1.machines) == 1

  print 'STARTING LONG PROCESS'
  long.start(long_comm)
  assert not long.poll()
  print 'STARTING SHORT PROCESSES'
  short0.start(short_comm0)
  short1.start(short_comm1)
  print 'TESTING PROCESS OVERLAP'
  assert not long.poll()
  print 'TESTED PROCESS OVERLAP'
  short0.wait()
  print 'FIRST SHORT PROCESS FINISHED'
  assert not long.poll()
  print 'TESTED PROCESS OVERLAP'
  short1.wait()
  print 'SECOND SHORT PROCESS FINISHED'
  assert not long.poll()
  print 'TESTED PROCESS OVERLAP'
  long.wait()
  print 'LONG PROCESS FINISHED'

  assert lfunc.Extract('long').success
  assert sfunc.Extract('short0').success
  assert sfunc.Extract('short1').success
  print 'END'
  

if __name__ == '__main__': 
  # ppn should be the number of processors per node
  # the job should be allocated with 2*ppn processors
  ppn = 32
  # path is the path to the Pylada source code
  path = '~/usr/src/Pylada/master/'

  from pylada.misc import RelativePath
  from os import chdir
  chdir('/home/e05/e05/mdavezac/bull') # crappy Cray
  path = RelativePath(path).path
  import sys
  sys.path.append(path + '/process/tests')
  print sys.path
  
  test( nbprocs=2*ppn, ppn=ppn,
        executable=path + '/build/process/tests/pifunc' )

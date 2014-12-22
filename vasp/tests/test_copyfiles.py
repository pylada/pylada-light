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

def test_copy_files():
  from tempfile import mkdtemp
  from shutil import rmtree
  from os import makedirs
  from os.path import exists, join
  from pylada.vasp import Vasp

  directory = mkdtemp()
  if exists(directory):
    rmtree(directory)
  makedirs(directory)
  makedirs(join(directory, 'indir'))
  with open(join(directory, 'indir', 'infile'), 'w') as file:
    file.write('hello')
  with open(join(directory, 'indir', 'this'), 'w') as file:
    file.write('hello')
  with open(join(directory, 'indir', 'that'), 'w') as file:
    file.write('hello')
  try:
    vasp = Vasp()
    # Shouldn't copy yet
    vasp._copy_additional_files(outdir=join(directory, 'outdir'))
    assert not exists(join(directory, 'outdir', 'infile'))
    assert not exists(join(directory, 'outdir', 'this'))
    assert not exists(join(directory, 'outdir', 'that'))
    # Now should copy
    vasp.files = join(directory, 'indir', 'infile')
    vasp._copy_additional_files(outdir=join(directory, 'outdir'))
    assert exists(join(directory, 'outdir', 'infile'))
    assert not exists(join(directory, 'outdir', 'this'))
    assert not exists(join(directory, 'outdir', 'that'))
    # Do it again, should be fine
    vasp._copy_additional_files(outdir=join(directory, 'outdir'))
    assert exists(join(directory, 'outdir', 'infile'))
    # Copy mutliple files
    vasp.files = [
      join(directory, 'indir', u) for u in ['infile', 'this', 'that']
    ]
    vasp._copy_additional_files(outdir=join(directory, 'outdir'))
    assert exists(join(directory, 'outdir', 'infile'))
    assert exists(join(directory, 'outdir', 'this'))
    assert exists(join(directory, 'outdir', 'that'))

  finally:
    if directory != '/tmp/test':
        rmtree(directory)


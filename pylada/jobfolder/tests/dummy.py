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


def Extract(outdir=None):
    from os.path import exists
    from os import getcwd
    from collections import namedtuple
    from pickle import load
    from pylada.misc import local_path

    if outdir == None:
        outdir = getcwd()

    Extract = namedtuple('Extract', ['success', 'directory', 'indiv', 'functional'])
    if not exists(outdir):
        return Extract(False, outdir, None, functional)

    outdir = local_path(outdir)
    if not outdir.join('OUTCAR').check(file=True):
        return Extract(False, str(outdir), None, functional)
    indiv, value = load(outdir.join('OUTCAR').open('rb'))

    return Extract(True, outdir, indiv, functional)


def functional(indiv, outdir=None, value=False, **kwargs):
    from pylada.misc import local_path
    from pickle import dump
    outdir = local_path(outdir)
    outdir.ensure(dir=True)
    dump((indiv, value), outdir.join('OUTCAR').open('wb'))
    return Extract(str(outdir))
functional.Extract = Extract

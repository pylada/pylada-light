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

""" Namespace for standard list of files to repatriate. """

POSCAR  = "POSCAR"
""" Name of the input structure file. """
KPOINTS = "KPOINTS"
""" Name of the kpoints file. """
INCAR   = "INCAR"
""" Name of the input parameters file. """
POTCAR  = "POTCAR"
""" Name of the pseudopotential file. """
WAVECAR = "WAVECAR"
""" Name of the wavefunction file. """
CONTCAR = "CONTCAR"
""" Name of the output structure file. """
CHGCAR  = "CHGCAR"
""" Name of the output charge file. """
OSZICAR = "OSZICAR"
""" Name of the energy minimization file. """
STDOUT  = "stdout"
""" Name of the standard output file. """
STDERR  = "stderr"
""" Name of the standard error file. """
EIGENVALUES = "EIGENVAL"
""" Name of the file with eigenvalues. """
OUTCAR = "OUTCAR"
""" Name of the output file. """
WAVEDER = "WAVEDER"
""" Name of GW-required file. """
TMPCAR = 'TMPCAR'
""" Name of temporary wavefunctions file. """
POT    = 'POT'
""" Name of the local potential file. """

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

CRYSTAL_geom_blocks = set(['CRYSTAL', 'SLAB', 'POLYMER', 'HELIX', 'MOLECULE', 'EXTERNAL'])
""" List of starting blocks in CRYSTAL input.

    CRYSTAL input does not differentiate between its block and keyword inputs.
    As such, to parse its input. the name of the blocks must be known
    explicitely. 
    This particular set is used to figure out where the input starts in an
    output file.
"""
CRYSTAL_input_blocks = set([ 'MARGINS', 'BIDIERD', 'CPHF', 'ELASTCON', 'EOS',
                             'SYMMWF', 'LOCALWF', 'ANISOTRO', 'ECH3', 'EDFT',
                             'EIGSHROT', 'OPTGEOM', 'FIXINDEX', 'GRID3D',
                             'MAPNET', 'POT3', 'DFT', 'PRINTOUT',
                             'REFLECTANCE', 'ROTCRY' ])                        \
                       | CRYSTAL_geom_blocks
""" List of blocks in CRYSTAL input.

    CRYSTAL input does not differentiate between its block and keyword inputs.
    As such, to parse its input. the name of the blocks must be known
    explicitely.
"""

CRYSTAL_filenames = { 'crystal.out':   '{0}.out',      # output file
                      'crystal.err':   '{0}.err',      # error file
                      'crystal.d12':   '{0}.d12',      # input file
                      'fort.9':        '{0}.f9',       # binary wave-functions
                      'fort.98':       '{0}.f98',      # formatted wave-functions
                      'GAUSSIAN.DAT':  '{0}.gjf',      # Gaussian 94/98 input
                      'MOLDRAW.DAT':   '{0}.mol',      # MOLDRAW input
                      'fort.33':       '{0}.xyz',      # xyz/Xmol input
                      'fort.34':       '{0}.gui',      # DLV input
                      'FINDSYM.DAT':   '{0}.findsym',  # Findsym input
                      'OPTHESS.DAT':   '{0}.opthess',  # formatted hessian
                      'OPTINFO.DAT':   '{0}.optinfo',  # restart info
                      'PPAN.DAT':      '{0}.ppan',     # Muliken population analysis
                      'ERROR':         '{0}.err',      # Error file in parallel CRYSTAL
                      'SCFOUT.LOG':    '{0}.scflog'    # SCF log
                    }
""" Filnames of crystal programs. 

    Map from fortran output filename to desired name. That desired format
    should always be the same to make it easy to find directories with CRYSTAL
    outputs.
""" 
CRYSTAL_propnames = { 'fort.25':      '{0}.f25',      # bands, maps, doss data
                      'GRED.DAT':     '{0}.GRED',     # direct lattice
                      'KIBZ.DAT':     '{0}.KIBZ',     # reciprocal lattice, IBZ
                      'KRED.DAT':     '{0}.KRED',     # reciprocal lattice, full BZ
                      'LINEA.DAT':    '{0}.LINEA',    # EMD line
                      'PROF.DAT':     '{0}.PROF',     # EMD in a plane
                      'DIEL.DAT':     '{0}.DIEL',     # dielectric constant
                      'POTC.DAT':     '{0}.POTC',     # exact electrostatic potential
                      'fort.31':      '{0}.prop3d',   # charge/spin density/potential
                      'fort.8':       '{0}.localwf',  # wannier function
                      'freqinfo.DAT': '{0}.freqinfo', # info for freq restart
                      'BAND.DAT':     '{0}.bands'     # band-structure info
                    }
CRYSTAL_delpatterns = ['core', 'ERROR.*']
""" Delete files with these patterns. 

    CRYSTAL leaves a lot of crap around. The patterns indicates files which
    should be removed. This is only applied if the working directory is the
    thesame as the output directory.
"""

def crystal_program(self=None, structure=None, comm=None):
  """ Path to serial or mpi or MPP crystal program version. 
  
      If comm is None, then returns the path to the serial CRYSTAL_ program.
      Otherwise, if :py:attr:`dftcrystal.Functional.mpp
      <pylada.dftcrystal.electronic.Electronic.mpp>` is
      True, then returns the path to the MPP version. If that is False, then
      returns the path to the MPI version.
  """
  ser = 'crystal'
  mpi = 'Pcrystal'
  mpp = 'MPPcrystal'
  if self is None or comm is None or comm['n'] == 1: return ser
  if self.mpp is True: return mpp
  return mpi

crystal_inplace = True
""" Wether to perform calculation in-place or in a tmpdir. """

properties_program = 'properties'
""" Path to single-electron CRYSTAL properties program. """

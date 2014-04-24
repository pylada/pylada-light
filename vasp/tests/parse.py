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

def test():
  from pylada.vasp import parse_incar
  from pylada.error import ValueError

  string = """ALGO = Fast\n"""\
           """ENCUT = 294.414\n"""\
           """EDIFF = 2e-12\n"""\
           """MAGMOM = -0*8 1.5*2\n"""\
           """ISPIN = 1\n"""\
           """ISMEAR = -1\n"""\
           """SIGMA = 0.001\n"""\
           """ISIF = 2\n"""\
           """NSW = 50\n"""\
           """IBRION = 2\n"""\
           """LMAXMIX = 4\n"""\
           """LCHARG = .TRUE.\n"""\
           """LVTOT = .FALSE.\n"""\
           """SYSTEM = Zinc-Blende\n"""


  def get_errors(found):
    errors = {}
    expected = { 'ALGO': 'Fast', 'ENCUT': '294.414', 'EDIFF': '2e-12',
                 'ISPIN': '1', 'MAGMOM': '0*8 1.5*2', 'ISMEAR': '-1',
                 'SIGMA': '0.001', 'ISIF': '2', 'NSW': '50', 'IBRION': '2',
                 'LMAXMIX': '4', 'LCHARG': '.TRUE.', 'LVTOT': '.FALSE',
                 'SYSTEM': 'Zinc-Blende' }
    for key in set(found.keys()) - set(expected.keys()): 
      errors[key] = found[key]
    for key in set(expected.keys()) - set(found.keys()): 
      errors[key] = None
    return errors
 
  result = parse_incar(string)
  assert len(get_errors(result)) == 0
  assert parse_incar(string.replace('\n', '\n#')) == [('ALGO', 'Fast')]
  assert len(get_errors(parse_incar(string.replace('\n', ';', 2)))) == 0
  assert len(get_errors(parse_incar(string.replace('=', '\\\n  =', 2)))) == 0
  assert len(get_errors(parse_incar(string.replace('=', '=  \\\n  ', 2)))) == 0

  try: parse_incar( string + "LVTOT = .TRUE.") 
  except ValueError: pass
  else: raise Exception()
  try: parse_incar( string + "   = .TRUE.") 
  except ValueError: pass
  else: raise Exception()

if __name__ == '__main__': 
  test()
   
    


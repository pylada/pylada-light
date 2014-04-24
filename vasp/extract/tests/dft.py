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

def test(path=None, dodate=True):
  from numpy import array, all, abs
  from pylada.vasp import Extract
  from quantities import eV, J, kbar as kB, angstrom

  a = Extract(directory=path)
  assert a.success == True
  assert a.energy_sigma0.units == eV
  try: assert a.energy_sigma0.units == J
  except: pass
  else: raise RuntimeError('test invalid')
  assert abs(a.energy_sigma0 - array(-10.661777) * eV) < 1e-8
  assert a.energies_sigma0.units == eV
  assert all(abs(a.energies_sigma0 -\
       array([ -1.40836383, -10.77001973, -10.85230124, -10.85263532,
               -10.85263539, -10.71541599, -10.65360429, -10.6553257 ,
               -10.65545068, -10.65547322, -10.65547247, -10.655472  ,
               -10.66729273, -10.66178619, -10.65881164, -10.65880468,
               -10.658805  , -10.66320019, -10.66223666, -10.66177273,
               -10.66177703, -10.661777  ]) * eV ) < 1e-8)
  assert a.all_total_energies.units == eV
  assert all(abs(a.all_total_energies\
        - array([ -1.41029646, -10.77396181, -10.85616651, -10.85650059,
                  -10.85650066, -10.71928126, -10.65746956, -10.65919097,
                  -10.65931595, -10.65933849, -10.65933774, -10.671158  ,
                  -10.66565146, -10.66267691, -10.66266995, -10.66706546,
                  -10.66610193, -10.66563801, -10.6656423 ]) * eV) < 1e-8)
  assert a.fermi0K.units == eV
  assert abs(a.fermi0K - 4.8932 * eV) < 1e-4
  assert not a.halfmetallic
  assert a.cbm.units == eV and abs(a.cbm - 7.1845*eV) < 1e-3
  assert a.vbm.units == eV and abs(a.vbm - 4.8932*eV) < 1e-3
  assert a.total_energy.units == eV and abs(a.total_energy + 10.665642*eV) < 1e-5
  assert a.total_energies.units == eV\
         and all(abs(a.total_energies - array([-10.659338, -10.66267 , -10.665642]) * eV) < 1e-5)
  assert a.fermi_energy.units == eV and abs(a.fermi_energy - 5.0616*eV) < 1e-4
  try: a.moment
  except: pass
  else: raise RuntimeError()
  assert a.pressures.units == kB and all(abs(a.pressures - array([ 21.96, -14.05,   0.1 ])*kB) < 1e-8)
  assert a.pressure.units == kB and abs(a.pressure - 0.1*kB) < 1e-8
  assert abs(a.alphabet + 11.7091) < 1e-4
  assert abs(a.xc_g0 + 9.2967) < 1e-4
  assert a.pulay_pressure.units == kB and abs(a.pulay_pressure) < 1e-8
  assert all(abs(a.recommended_fft - 21) < 1e-8)
  assert all(abs(a.fft - 20) < 1e-8)
  assert a.partial_charges is None
  assert a.magnetization is None
  assert a.eigenvalues.units == eV\
         and all(abs(a.eigenvalues \
         - [[-5.3915, 1.7486, 4.8932, 4.8932, 7.5932, 9.1265, 9.1265, 12.3814],
            [-3.5142, -0.6886, 2.1711, 3.4739, 7.1845, 10.0803, 11.2883,  11.5243]] * eV) < 1e-8)
  assert all(abs(a.occupations - [[ 2.,  2.,  2.,  2., -0., -0., -0.,  0.],
                                  [ 2.,  2.,  2.,  2., -0., -0.,  0.,  0.]]) < 1e-8)
  assert a.electropot.units == eV and all(abs(a.electropot - [-83.4534, -83.4534]*eV) < 1e-8)
  assert a.forces.units == eV/angstrom and all(abs(a.forces) < 1e-8)
  assert a.stresses.units == kB \
         and all(abs(a.stresses - [[[ 21.96,   0.  ,   0.  ],
                                    [  0.  ,  21.96,   0.  ],
                                    [  0.  ,   0.  ,  21.96]],
                                   [[-14.05,   0.  ,   0.  ],
                                    [  0.  , -14.05,   0.  ],
                                    [  0.  ,   0.  , -14.05]],
                                   [[  0.1 ,   0.  ,   0.  ],
                                    [  0.  ,   0.1 ,   0.  ],
                                    [  0.  ,   0.  ,   0.1 ]]] * kB) < 1e-3)
  assert a.stress.units == kB \
         and all(abs(a.stress - [[  0.1 ,   0.  ,   0.  ],
                                 [  0.  ,   0.1 ,   0.  ],
                                 [  0.  ,   0.  ,   0.1 ]] * kB) < 1e-3)
  assert hasattr(a.structure, 'stress')\
         and a.structure.stress.units == a.stress.units \
         and all(abs(a.structure.stress - a.stress) < 1e-8)
  assert all([hasattr(b, 'force') for b in a.structure])\
         and all([b.force.units == a.forces.units for b in a.structure])\
         and all(abs(a.forces.magnitude - array([b.force for b in a.structure])) < 1e-8)

if __name__ == "__main__":
  from sys import argv, path 
  from os.path import join
  if len(argv) > 2: path.extend(argv[2:])
  
  test(join(argv[1], 'COMMON'))


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

from pylada.vasp.specie import nlep

def_rng = 5
def_rng3 = 5
def_small_rng = 4
def_large_rng = 20

nlep_Al_s = nlep(type="Dudarev", l="s", U0=1, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Al_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
#this way, can also be specified as:
#nlep_Al_s = {'U1': 0.0, 'l': 0, 'U0': 1.0, 'U1_range': 0.69999999999999996, 'func': 'enlep', 'U0_range': 0.5, 'fitU1': True, 'fitU0': True, 'type': 2}
#direct from results of previous run.

nlep_Mg_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Mg_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)

nlep_Ga_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Ga_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Ga_d = nlep(type="Dudarev", l="d", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)

nlep_In_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_In_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_In_d = nlep(type="Dudarev", l="d", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)

nlep_Zn_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Zn_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Zn_d = nlep(type="Dudarev", l="d", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)

nlep_Cd_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Cd_p = nlep(type="Dudarev", l="p", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_Cd_d = nlep(type="Dudarev", l="d", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)

# special treatment for the anions:
nlep_N_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_rng)
nlep_N_p = nlep(type="Dudarev", l="p", U0=0, U1=20, fitU0=True, fitU1=False, U0_range=def_rng,  U1_range=def_rng)
    
nlep_O_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_large_rng)
nlep_O_p = nlep(type="Dudarev", l="p", U0=-1.9, U1=20, fitU0=True, fitU1=False, U0_range=def_large_rng,  U1_range=def_rng)
    
nlep_P_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_small_rng)
nlep_P_p = nlep(type="Dudarev", l="p", U0=0, U1=10, fitU0=True, fitU1=False, U0_range=def_rng,  U1_range=def_small_rng)
                      
nlep_S_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng,  U1_range=def_small_rng)
nlep_S_p = nlep(type="Dudarev", l="p", U0=0, U1=10, fitU0=True, fitU1=False, U0_range=def_rng,  U1_range=def_small_rng)
    
nlep_As_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng3,  U1_range=def_rng3)
nlep_As_p = nlep(type="Dudarev", l="p", U0=0, U1=5, fitU0=True, fitU1=False, U0_range=def_rng3,  U1_range=def_rng3)
                      
nlep_Se_s = nlep(type="Dudarev", l="s", U0=0, U1=0, fitU0=True, fitU1=True, U0_range=def_rng3,  U1_range=def_rng3)
nlep_Se_p = nlep(type="Dudarev", l="p", U0=0, U1=5, fitU0=True, fitU1=False, U0_range=def_rng3,  U1_range=def_rng3)
    
nlep_element_defaults = {'Al':[nlep_Al_s, nlep_Al_p], 'Ga':[nlep_Ga_s, nlep_Ga_p, nlep_Ga_d],  'In':[nlep_In_s, nlep_In_p, nlep_In_d],
                 'N':[nlep_N_s, nlep_N_p], 'P':[nlep_P_s, nlep_P_p],  'As':[nlep_As_s, nlep_As_p],
                 'Mg':[nlep_Mg_s, nlep_Mg_p], 'Zn':[nlep_Zn_s, nlep_Zn_p, nlep_Zn_d],  'Cd':[nlep_Cd_s, nlep_Cd_p, nlep_Cd_d],
                 'O':[nlep_O_s, nlep_O_p], 'S':[nlep_S_s, nlep_S_p],  'Se':[nlep_Se_s, nlep_Se_p]}





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

""" NLEP fitting input script. """
from os.path import join, exists
from shutil import rmtree
import numpy as np

from pylada.vasp import Extract
from pylada.vasp import Vasp
from pylada.vasp.nlep.nlep_defaults import nlep_element_defaults


class SystemParams:
    def __init__(self, cation, anion, potcar_dir, outcar_data_dir, floating_vbm, nlep_params = None):
        # cation, anion are strings
        self.species_names = [cation,anion]
        self.cation = self.species_names[0]
        self.anion = self.species_names[1]
        self.bounds = None
        self.scale_pressure = None  # this leaves it to global default in input_generic.py or run_input.py; otherwise set it to sys-specific value

        self.atom_idx = [0,1]
        """total hack that maps how pylada happens to enumerate the atoms to how the
        dft fitting data orders the atoms. TODO. a solution, rsises to "workaround",
        is to have functions to probe for and establish the connection"""
        # eg probe how python dict will order the species, then possibly
        # reorder atom indexing in partial charge code
#testdict = {'<cation>':0, '<anion>':0}
#first = testdict.popitem()[0]
#print "first = ", first
#if first == '<anion>':
#    atom_idx = [1,0]
#    print "anion <anion> is first by python"
# This doesn't work.  e.g. AsAl comes out, but Pylada gives AlAs.  Seems alphabetic for 3-5 systems, so I
# will just make that happen:
        if anion < cation:
            self.atom_idx = [1,0]
            print "%s < %s, swapping order" % (anion, cation)
        else:
            print "%s < %s, order kept" % (cation ,anion)
    
        self.special_SSe = False
        self.special_cbm_gap = False
        self.orig_weights = False
        self.with_d = False
        self.cation_potcar_ext = ""
        self.anion_potcar_ext = ""
        if (cation in ["Ga","In","Zn","Cd"]):
            self.with_d = True
        if (cation in ["Ga","In"]):
            self.cation_potcar_ext = "_d"
        if (anion in ["O","N"]):
            self.anion_potcar_ext = "_s"

        self.cbm_gap_state_idx = 0  # default is transition to Gamma
        if (cation == "Al"):
            self.cbm_gap_state_idx = 1  # X
        if (cation == "Ga" and anion == "P"):
            self.cbm_gap_state_idx = 1  # X
        if (cation == "Mg" and anion == "S"):
            self.cbm_gap_state_idx = 1  # X
        if (cation == "Mg" and anion == "Se"):
            self.cbm_gap_state_idx = 1  # X


        if (self.with_d):
            self.nlep_band_idx = [1,6,7,8,9,10]   # custom states ala SL.
            """ eigs (indices) to consider in obj fn, vasp-nlep data """
    ### for setting up offsets to reference everything from VBM
            self.nlep_valence_band_idx = 8
            """ which band is the valence band  (probably does not need to be different than gw_valence_band_idx; from bug)"""
            if self.orig_weights:
                self.eigenvalue_weights = np.array(\
                    [  [    0.2,     0.5,  0,    0,   1,   0.5    ],
                       [    0,       0.5,  0,    1,   1,   0.5    ], 
                       [    0,       0.5,  0,    1,   1,   0.5    ]])
                self.eigenvalue_weights[cbm_gap_state_idx][4] += 1
            else:
                self.eigenvalue_weights = np.array(\
                    [  [    0.2,     0.5,  0,    0,   1.5,   0.5    ],
                       [    0,       0.5,  0,    1,   2.0,   1.0    ], 
                       [    0,       0.5,  0,    1,   1,   0.5    ]])
            """ band weights,# a len kidx X len bandidx array
            k down, band across 
            ##     deep-d   VBM3  VBM2  VBM1  CBM1 CBM2   
            # G:
            # X:
            # L:
            rules: subject to exceptions, VBM1 and CBM1 = 1, VBM2 and CBM2 = 0.5.
            exceptions: VBM at G=0 b/c it is the reference energy
            CBM that determeines gap = 2 (usually Gamma)
            For anions with d-bands, fit deep d-state with weight = 0.2
            """
            if (floating_vbm):
                self.eigenvalue_weights[0][3] = 1
    
            if (self.special_SSe):
                if (self.anion in ["S"]):
                    print "special weights for S CB-X state"
                    self.eigenvalue_weights[1][4] *= 2.0 
                elif (self.anion in ["Se"]):
                    print "special weights for Se CB-X state"
                self.eigenvalue_weights[1][4] *= 1.5 

            self.pc_orbital_weights = np.array(\
                [        [3, 3, 2],
                         [3, 1.5, 0]])
            """ orbital weights
            # a num_atoms X num_orbitals array
            ##       s   p   d
            #cation
            #anion
            rules: all get weight 3 except for anion-p and cation-d, which get 2
            """
        else:
            self.nlep_band_idx = [1,2,3,4,5]   # custom states ala SL.
            ### for setting up offsets to reference everything from VBM
            ### this is index of this state in the OUTCAR.
            self.nlep_valence_band_idx = 3

            if self.orig_weights:
                self.eigenvalue_weights = np.array(\
                [  [    0.5,  0,    0,   1,   0.5    ],
                   [    0.5,  0,    1,   1,   0.5    ], 
                   [    0.5,  0,    1,   1,   0.5    ]])
                self.eigenvalue_weights[cbm_gap_state_idx][3] += 1
            else:
                self.eigenvalue_weights = np.array(\
                [  [    0.5,  0,    0,   1.5,   0.5    ],
                   [    0.5,  0,    1,   2.0,   1.0    ], 
                   [    0.5,  0,    1,   1,   0.5    ]])
            """ band weights,# a len kidx X len bandidx array
            k down, band across 
            ##      VBM3  VBM2  VBM1  CBM1 CBM2   
            # G:
            # X:
            # L:
            """
            if (floating_vbm):
                self.eigenvalue_weights[0][2] = 1
            if (self.special_SSe):
                if (self.anion in ["S"]):
                    print "special weights for S CB-X state"
                    self.eigenvalue_weights[1][3] *= 2.0 
                if (self.anion in ["Se"]):
                    print "special weights for Se CB-X state"
                    self.eigenvalue_weights[1][3] *= 1.5 

            self.pc_orbital_weights = np.array(\
                [        [3, 3, 0],
                         [3, 1.5, 0]])
            """ orbital weights
            # a num_atoms X num_orbitals array
            ##       s   p   d
            #cation
            #anion
            rules: all get weight 3 except for anion-p and cation-d, which get 2
            """
    
        self.gw_band_idx = self.nlep_band_idx  # 4 states around gap ?
        """ eigs (indices) to consider in obj fn, in corresponding gw data """


        self.gw_valence_band_idx = self.nlep_valence_band_idx
        """ which band is the valence band in the gw data"""


        # special adjustment b/c Mg gw data uses different POTCAR than we are using for fitting.
        if (self.cation == "Mg"):
            self.gw_band_idx = [4,5,6,7,8]
            self.gw_valence_band_idx = 6

        # k-point   1 :       0.0000    0.0000    0.0000
        #   band No.  band energies     occupation
        #         1     -12.2303      2.00000
        #         2       4.5371      2.00000
        #         3       4.5371      2.00000
        #         4       4.5371      2.00000
        #         5       9.8453      0.00000
        #         6      21.0976      0.00000
        #         7      21.0976      0.00000
        #         8      21.0976      0.00000


        #k-point   1 :       0.0000    0.0000    0.0000
        #band No. DFT-energies  QP-energies   sigma(DFT)  V_xc(DFT)    V^pw_x(r,r')   Z            occupation
        #
        #1     -44.0573     -44.0817     -34.9956     -34.9673     -40.9513       0.8637       2.0000
        #2     -44.0573     -44.0817     -34.9956     -34.9673     -40.9513       0.8637       2.0000
        #3     -44.0573     -44.0817     -34.9956     -34.9673     -40.9513       0.8637       2.0000
        #4     -16.3055     -16.3452     -21.7455     -21.6937     -29.9507       0.7656       2.0000
        #5       2.6840       2.6679     -21.5656     -21.5459     -25.2449       0.8184       2.0000
        #6       2.6840       2.6679     -21.5656     -21.5459     -25.2449       0.8184       2.0000
        #7       2.6840       2.6679     -21.5656     -21.5459     -25.2449       0.8184       2.0000
        #8      10.4955      10.5027     -11.4741     -11.4823      -8.0981       0.8677       0.0000
        #9      22.0443      22.0585      -7.8777      -7.8942      -3.2856       0.8612       0.0000


        #dft_in = Extract(directory="/home/pagraf/projects/pylada/nlep/data/GW-26-35/")
        #dft_in = Extract(directory="/scratch/pagraf/projects/pylada/nlep/data/")
        """ Object which extracts DFT output """
        #dft_in.OUTCAR = "<cation><anion>_OUTCAR_pbe" # is "OUTCAR" by default
        #dft_in.CONTCAR = "<cation><anion>_POSCAR"    # is "CONTCAR" by default

        self.dft_in = Extract("%s/%s%s_OUTCAR_pbe" % (outcar_data_dir, cation, anion))

        #gw_in = ExtractGW(directory="/home/pagraf/projects/pylada/nlep/data/GW-26-35")
        #gw_in = ExtractGW(directory="/scratch/pagraf/projects/pylada/nlep/data/")
        """ Object which extracts GW output """
        #gw_in.OUTCAR = "<cation><anion>_OUTCAR_gw"  # is "OUTCAR" by default
        #gw_in.CONTCAR = "<cation><anion>_POSCAR"    # is "CONTCAR" by default

        self.gw_in = Extract("%s/%s%s_OUTCAR_gw" % (outcar_data_dir, cation, anion))

        self.vasp = Vasp\
               (
                 kpoints    = "\n0\ngamma\n8 8 8\n0 0 0",
                 precision  = "accurate",
                 ediff      = 1e-5,
                 encut      = 1.3, # uses ENMAX * 1, which is VASP default
                 nbands     = 16,
                 lorbit     = 10,
                 npar       = 2,
                 lplance    = True,
                 addgrid    = True,
                 relaxation = "ionic",
        #         set_smearing   = "bloechl",
                 set_smearing   = "0",
                 sigma = 0.001,
                 ismear = 0 # gaussian
               )
        """ VASP functional """

        if (nlep_params != None and cation in nlep_params):
            self.vasp.add_specie = "%s" % cation, "%s/%s%s" % (potcar_dir, cation, self.cation_potcar_ext), nlep_params[cation]
        else:
            self.vasp.add_specie = "%s" % cation, "%s/%s%s" % (potcar_dir, cation, self.cation_potcar_ext), nlep_element_defaults[cation]

        if (nlep_params != None and anion in nlep_params):
            self.vasp.add_specie = "%s" % anion, "%s/%s%s" % (potcar_dir, anion, self.anion_potcar_ext), nlep_params[anion]
        else:
            self.vasp.add_specie = "%s" % anion, "%s/%s%s" % (potcar_dir, anion, self.anion_potcar_ext), nlep_element_defaults[anion]

        if (anion == "O"):
            self.bounds = [-20,20]

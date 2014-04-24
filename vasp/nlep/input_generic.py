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

outdir = "nlep_fit"
""" Saves VASP calculation in this directory. """

nbjobs = -1
""" Number of different fitting jobs to perform.

    Jobs are independent and start from different random starting points. 
    Each job will have access to ~world.size/nbjobs. 
    Will throw if world.size/nbjobs < 0
    nbjobs = -1 used to say: do as many jobs as there are processors
"""

max_functional_eval = 5000
""" Maximum number of VASP evaluations.
    Mimimizer should stop after reaching this point.
"""


k_idx = [0,4,20]  # On Stephans suggestion, just consider G,X,L 
""" which k points to consider """
gamma_point_idx = 0
""" which k is the gamma point"""

#k_idx = [0,9,50,59]  # on Mayeul's rec!
#k_idx = [0,5,12,17,20,2,3,4,5]  # these are g2x and g2l lines 


pc_orbital_idx = [0,1,2]  # s and p and d, but can select
""" which orbitals' (specify list of indices) partial charges we care about"""


step_factor = 10
""" Initial step factor. See scipy.minpack.leastsq. """
xtol = 1e-10
""" Relative error desired in the approximate solution. """
ftol = 1e-10
""" Relative error desired in the sum of squares. """
epsfcn=0.001
""" steplength for finite difference calculations"""




x0 = None
#x0 = [[0,0,0,0,  0,0,0,0]]
""" initial nlep parameters (and/or center of scans)-- overrides initial values in add_specie ... below
The list need not be as long as the number of processors.  The rest will be initialized randomly within the bounds"""

scan_half_range = 2
#scan_params = [[x0[0]-scan_half_range, x0[0]+scan_half_range, 0],[x0[1]-scan_half_range, x0[1]+scan_half_range, 1], [x0[2]-scan_half_range, x0[2]+scan_half_range, 2]]
#xx0 = x0[0]
xx0 = [0,0,0,0,  0,0,0,0]
scan_params = [[xx0[0]-scan_half_range, xx0[0]+scan_half_range, 0]]
"""scan parameters, list of [<pmin><pmax><param idx>]"""
scan_steps = 5

########  what to do:
one_shot = False
""" True => just do vasp runs for each element of x0"""

do_scan = False
""" True=> don't optimize, just do a parameter scan. will perform series of 1D scans"""

do_lsq = True
""" True => scipy.optimize.leastsq"""

do_lmmin = False
""" True => lmmin, parallel least squares"""

#########

bounds = [-5, 5]
""" Range within which to choose random start vector. """

bounds_penalty = 1e4
""" Strength to enforce out of bounds penalty"""

units = 1e0
""" Scales all nlep parameters by this number. """

#scale_eigenvalues = 1e0/5e0
scale_eigenvalues = 1e0
""" Scales eigenvalues by this factor.

    If M{S{epsilon}(n,k)} are the eigenvalue, then they enter the least square
    sum as: M{S{sum} ||S{lambda}S{epsilon}(n,k)||^2}. If zero, then eigenvalues
    do not enter the fit.
    replaced by 'eigenvalue_weights' 
"""
#scale_partial_charges = 300e0
scale_partial_charges = 1e0
""" Scales partial_charges by this factor.
replaced by 'orbital_weights'
"""

## rough reasoning:  want eigenvalues w/in 0.1 eV,  want pressure w/in 10 kB.
## code already scales eigenvalue contribution by how many there are, so we want
## the ratio of eigenvalue scale to pressure scale to be pmin/emin ~ 100
## Similarly, want partial charges w/in 0.01 to 0.1 electrons, so ratio of
## eval scale to pc scale shoule be pcmin/emin ~ [.01 --- .1]/.1 = [.1--1] = say 1

### Stephan Lany has given more specific suggestions, now implemented 

## added this before learning how to control is with smear and sigma
scale_occupations = 0

#scale_pressure = 1e0
#scale_pressure = 0.03
scale_pressure = 0.06
""" Scales pressure by this factor. """


vasp = None
dft_in = None
gw_in = None
nlep_params = None
random_start = None
# just so it exists ?
## this was due to bug, I think (and, to do with .pyc ?)##

### job specific ? #####
## moved here so project level input, not just system level input,
## knows what the job is.

optimizer = "leastsq"  # from scipy
# vs
# optimizer = "lmmin" # parallelized/pythonized (by PG) LM solver
nbjobs = 1
# can be run with mpirun -np 4 python popt,py


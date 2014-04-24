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

# how should this go?
# should have pylada-parsed version of all VASP data.
# instead I have my own script that parses OUTCAR and
# draws plots
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import sys

import plotbs
from run_input import cations, anions

def main():
    # this is really just a custom trigger to plot S. Lany's data with "MgOx" naming
    four_char_names = False
    plot_dft = False

    if (len(sys.argv) < 3):
        print "plot_fig <dir> <base OUTCAR name> [-fcn]"
        print "-fcn triggers 4 character names"
        sys.exit()

    total_dir = sys.argv[1]
    name = sys.argv[2]

    # support for some special data sets
    plotting_dft = False
    if (len(sys.argv) > 3):
        four_char_names = sys.argv[3] == "-fcn"
        plotting_dft = sys.argv[3] == "-dft"

    def map_ion(ion):
        if (not four_char_names):
            return ion
        elif (len(ion) != 1):
            return ion
        elif ion == "O":
            return "Ox"
        elif ion == "S":
            return "Su"

        else:
            return ion


    ncat = len(cations)
    nan = len(anions)
    basedir = "/Users/pgraf/work/cid/pylada/nlep/"
    subdir1 = "/nlep_materials_from_slany"
    name1 = "OUTCAR_gw"
    #if (four_char_names):
    #    subdir = "/nlep_materials_from_slany/slany_2-6/"
    #    name = "OUTCAR-single"
    #else:
    #    subdir = "/opt/from_redmesa/all_2-6_leastsq/best_fit/"
    #    name = "OUTCAR"
    #if (plot_dft):
    #    subdir = "/nlep_materials_from_slany"
    #    name = "OUTCAR_pbe"


    ifig = 1
    fig = plt.figure(1)
    for icat in range(0,ncat):
        cat = cations[icat]
        for ian in range(0,nan):
            an = anions[ian]

            vb_idx_nlep = 3
            bend_idx_nlep = 7
            if (cat in ["Ga","In","Zn","Cd"]):
                vb_idx_nlep = 8
                bend_idx_nlep = 12
            vb_idx_gw = vb_idx_nlep
            bend_idx_gw = bend_idx_nlep
            if (cat == "Mg"):
                vb_idx_gw = 6
                bend_idx_gw = 12
                if (plotting_dft):
                    vb_idx_nlep = 6
                    bend_idx_nlep = 12
            an1=an
            cat1=cat
            an = map_ion(an)  # for Stephans 4 character naming
            cat = map_ion(cat)
            args = "proc1.py --skipln --bend=%d %s/%s/%s%s_%s   --matchvbm=%d,%d --bend=%d %s/%s%s_%s"  % (bend_idx_gw, basedir, subdir1, cat1, an1, name1, vb_idx_gw, vb_idx_nlep, bend_idx_nlep, total_dir, cat, an, name)
            print args
            args = args.split()
    #        plt.subplot(ncat, nan, ifig)
            ax = fig.add_subplot(ncat, nan, ifig)
            fig = plotbs.real_main(args, fig, ax, ifig)
            tit = "%s%s" % (cat, an)
            fig.text(.2 + ian/3., 0.9 - icat/3., tit)
            fig.canvas.set_window_title("black = %s, red = %s,%s" % ( name1, total_dir, name))
            ifig += 1

    plt.show()


if __name__ == '__main__':
    main()






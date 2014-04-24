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

# simply process OUTCAR to do basic band structure diagram

import os, sys, optparse
import matplotlib 
matplotlib.use('TkAgg') 
#from pylab import plot, show
import matplotlib.pyplot as plt

from optparse import OptionParser

# from
#grep -B 1 -A 3 "^  band No" OUTCAR | awk '{if (NR%6==1) s=$0; else if (NR%6==4) printf "%s %s\n", s, $0}' > log2

# indices of kpoints to extract from gw data
gw_g2x = [0,5,12,17,20]
gw_g2l = [0,1,2,3,4] 
gamma_idx = 0

class BandStructure :
    def __init__(self, kg2x, g2x, kg2l, g2l):
        self.kg2x = kg2x
        self.kg2l = kg2l 
        self.g2x = g2x
        self.g2l = g2l

def split1(fn, ax, col, skipln, gwk,  first_vband_idx, my_vband_idx, bstart, bend, first_bs):
#fn = "OUTCAR"
    f = file(fn)
    ln = f.readline()
    all_kpt = []
    nband = -1

    while (ln != ""):
        stuff = ln.split()
    #    print stuff
        if (len(stuff)>1 and stuff[0] == "band" and stuff[1] == "No."):
            if (skipln == True):  # gw data has extra space after "band No."
                f.readline()
            ll = lastLn.split()
            ib = 0
            kpt_dat = []
            ln = f.readline().split()
            while (len(ln) > 0):
                band = [int(ln[0]), float(ln[1]), float(ln[2])]
                kpt_dat.append(band)
                kpt = [int(ll[1]), float(ll[3]), float(ll[4]), float(ll[5]), kpt_dat]
                ib +=1 
                ln = f.readline().split()

            all_kpt.append(kpt)
            if (nband > 0 and nband != ib):
                print "problem: found different number bands at different k points"
            nband = ib
        lastLn = ln
        ln = f.readline()

#    for kpt in all_kpt:
#        print kpt[0]-1, kpt[1], kpt[2], kpt[3]
    #    for b in kpt[4]:  # which is actually a list of energies for each band
            #print b
    #    print "%f %f %f    %f %f" % (kpt[1], kpt[2], kpt[3],  kpt[4][3][1], kpt[4][4][1])


    kg2x = []
    kg2l = []
    g2x = []
    g2l = []

    if (gwk == True):
        g2x_idx = gw_g2x
        g2l_idx = gw_g2l
    else:
        g2x_idx = range(0,10)
        g2l_idx = range(10,20)
        
    for b in range(0,nband):
        g2x.append([])
        g2l.append([])
    
    for i in g2x_idx:
        kx = all_kpt[i]
        kg2x.append(kx[1])
        kxdat = kx[4]    
        for b in range(0,nband):
            g2x[b].append(kxdat[b][1])

    for i in g2l_idx:
        kl = all_kpt[i]
        kg2l.append(-kl[1])
        kldat = kl[4]
        for b in range(0,nband):
            g2l[b].append(kldat[b][1])

#    for k in kg2x:
#        print k
#    print
#    for k in kg2l:
#        print k
    #    print kg2l[i]
#    for b in range(0,nband):
#        print "   ", g2x[b][0]

    #from numpy import array
    #gamma->X is first 10, gamma->L is 2nd 10
    offset = 0
    if (my_vband_idx != None and first_bs != None):
        offset = first_bs.g2x[first_vband_idx][gamma_idx] - g2x[my_vband_idx][gamma_idx]
    if (bstart == None):
        bstart = 0
    if (bend == None):
        bend = nband
    print bstart, bend, offset
    for b in range(bstart, bend):
        vals = [e + offset for e in g2x[b]]
        ax.plot(kg2x, vals, color=col)
        vals = [e + offset for e in g2l[b]]
        ax.plot(kg2l, vals, color=col)
    
    return BandStructure(kg2x, g2x, kg2l, g2l)


def real_main(argv, fig, ax, ifig = 0):
    if (len(argv) == 1):
        print "proc1.py <[options] band structure file>*"
        print "bs file is actual OUTCAR"
        print "options: --notgwk. WITHOUT this, we assume kpoints of interest are as in Stephan Lany's GW data"
        print "--notgwk will assume a run of only g2x (kpoints 0-9) and g2l (kpoint 10-19)"
        print "--skipln parses OUTCAR to match GW data. necessary for plotting bs from fitting data"
        print "--matchvbm=<n1><n2> says band n2 for this data file is same as band n1 for the first data file given"
        print "--bstart=<n1>  and --bend=<n2>  plots only bands from n1 to n2.  These options stay in effect until overridden"
        print "eg:"
        sys.exit()

#    fig = plt.figure(ifig)
#    ax = fig.add_subplot(1,1,1)
    cols = ['k','r','b','g']
    colidx = 0
    gwk = True
    skipln = False
    first_bs = None
    bstart = None
    bend = None
    my_vband_idx = None
    first_vband_idx = None
    for i in range(1,len(argv)):
        fn = argv[i]
        print fn
        if (fn == "--notgwk"):
            gwk = False
        elif (fn == "--skipln"):
            skipln = True
        elif (fn[0:10] == "--matchvbm"):
            #expecting "--matchvbm=<first vb idx>,<this vb idx>"
            stuff = fn[11:len(fn)].split(",")
            if (len(stuff) != 2):
                print "cannot parse %s", fn
            my_vband_idx = int(stuff[1])
            first_vband_idx = int(stuff[0])
        elif (fn[0:8] == "--bstart"):
            bstart = 1 + int(fn[9:len(fn)])  # one for "less than" indexing in "range()"
        elif (fn[0:6] == "--bend"):
            bend = int(fn[7:len(fn)])
        else:
            col = cols[colidx % (len(cols))]
            bs = split1(fn, ax, col, skipln, gwk, first_vband_idx, my_vband_idx, bstart, bend, first_bs)
            if (first_bs == None):
                first_bs = bs
            colidx += 1
            gwk = True
            skipln = False
            my_vband_idx = None
            first_vband_idx = None
    axis = ax.xaxis
    axis.set_ticks([-0.5, 0, 0.5])
    axis.set_ticklabels(["L","G","X"])

    return fig
#    return plt


def main():
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    real_main(sys.argv, fig, ax, 1)
    plt.show()
    
if __name__ == '__main__':
    main()

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

""" Extracts defect properties from many calculations. """
__docformat__ = "restructuredtext en"
from pylada import try_import_matplotlib
if try_import_matplotlib:
  __all__ = ['Single', 'Material', 'plot_enthalpies']

  from _single import Single
  from _mass import Material

  colors = "green red blue purple orange darksalmon darkgrey chocolate slategray burlywood".split()
  try: import matplotlib.pyplot as plt 
  except: 
    def plot_enthalpies(material, mu=None, **kwargs):
      """ Plots formation enthalpies versus Fermi energy using matplotlib. """
      print "Matplotlib does not seem to be running."
    def plot_transitions(material, **kwargs):
      print "Matplotlib does not seem to be running."
  else: 
    def plot_enthalpies(material, mu=None, **kwargs):
      """ Plots formation enthalpies versus Fermi energy using matplotlib. """
      from quantities import eV
      from operator import itemgetter

      _colors = kwargs.pop("colors", colors)
      
      # sets up some stuff for legends.
      plt.rc('text', usetex=True)
      plt.rc('text.latex', preamble="\usepackage{amssymb}")
      # finds limits of figure
      xlim = float(material.host.vbm.rescale(eV)), float(material.host.cbm.rescale(eV)) 
      all_ys = [ float(val.rescale(eV)) for x in xlim \
                 for val in material.enthalpies(x, mu).itervalues() ]
      ylim = min(all_ys), max(all_ys)

      # loop over defects.
      for i, defect in enumerate(material.ordered_values()):
        # finds intersection points. 
        x = [(xlim[0]-5e0)*eV]
        lines = defect.lines()
        for j in range(len(lines)-1):
          (b0, a0), (b1, a1) = lines[j], lines[j+1]
          x.append( ((b0-b1)/(a1-a0)).rescale(eV) )
        x.append((xlim[1]+5e0)*eV)

        # Now draws lines. 
        lines.append(lines[-1])
        if   defect.is_interstitial: linestyle, color =  '-', _colors[i % len(_colors)]
        elif defect.is_substitution: linestyle, color = '--', _colors[i % len(_colors)]
        elif defect.is_vacancy:      linestyle, color = '-.', _colors[i % len(_colors)]
        else:                        linestyle, color =  ':', _colors[i % len(_colors)]
        y = [u[0] + u[1] * xx for u, xx in zip(lines, x)]
        plt.plot(x, y, label=defect.latex_label, color=color, linestyle=linestyle, **kwargs)

      # plot vbm and cbm
      plt.axvline(material.vbm, color='black')
      plt.axvline(material.cbm, color='black')

      plt.legend()
      ylim = ylim[0] - (ylim[1]-ylim[0]) * 0.05, ylim[1] + (ylim[1]-ylim[0]) * 0.05
      xlim = xlim[0] - (xlim[1]-xlim[0]) * 0.05, xlim[1] + (xlim[1]-xlim[0]) * 0.05
      plt.xlim(xlim)
      plt.ylim(ylim)
      plt.xlabel("Fermi Energy [in eV]")
      plt.ylabel("$\Delta H_{D,q}(E_F)$ [in eV]")
      plt.draw()

    def plot_transitions(material, mu=None, **kwargs):
      """ Plots transition energies using matplotlib. """
      from quantities import eV
      from operator import itemgetter

      _colors = kwargs.pop("colors", colors)

      # sets up some stuff for legends.
      plt.rc('text', usetex=True)
      plt.rc('text.latex', preamble="\usepackage{amssymb}")

      xlim = (-0.5, len(material.jobs)-0.5)
      ylim = [material.vbm, material.cbm]

      mincharge = min(c for d in material.ordered_values() for c in d.charges)

      # loop over defects.
      for i, defect in enumerate(material.ordered_values()):
        # loop over charges
        charges = defect.charges
        for qi, qf in zip(charges[:-1], charges[1:]):
          e = defect.transition(qi, qf)
          if e < ylim[0]: ylim[0] = e
          if e > ylim[1]: ylim[1] = e
          c = int(qi - mincharge + 0.01) 
          plt.plot([i-.4, i+.4], [e, e], color=_colors[c%len(_colors)], **kwargs)
      
      # plot vbm and cbm
      plt.axhline(material.vbm, color='black')
      plt.axhline(material.cbm, color='black')


      plt.gca().set_xticks(range(len(material.jobs)), minor=False)
      xlbs = [a.latex_label for a in material.ordered_values() ]
      plt.gca().set_xticklabels(xlbs)
      plt.xlim(xlim)
      ylim = ylim[0] - (ylim[1]-ylim[0]) * 0.05, ylim[1] + (ylim[1]-ylim[0]) * 0.05
      plt.ylim(ylim)
      plt.ylabel("$\epsilon_{D,q}(q/q+1)$ [in eV]")
      plt.draw()

    # adds methods to class Material.
    Material.plot_enthalpies = plot_enthalpies
    Material.plot_transitions = plot_transitions
else: __all__ = []

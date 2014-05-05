.. _vasp_relax_ug:

A *meta*-functional: Strain relaxation
======================================

Relaxing structures generally takes a few actual VASP calculations, since the
FFT and pseudo-potential grids are not adapted to changing cell parameters and
ionic positions. It's brain-dead work best handled automatically. Pylada
currently provides two relaxation methods: :py:class:`relax.Relax` and
:py:class:`relax.Epitaxial`. The former handles general relaxation, including
cellshape, volume, and ionic positions, while the latter performs relaxation on
a virtual substrate (e.g. for coherent Germanium on a (001)@Si substrate, only
the out-of-plane parameter should be relaxed, while the in-plane are fixed to
Si).  The following only describes the first case.

>>> # create the functional
>>> from pylada.vasp import Relax
>>> functional = Relax(relaxation='cellshape volume ions', maxcalls=10, keepsteps=True)
>>> functional(structure)

The above creates the functional and launches the calculations. It will first
proceed by relaxing everything, e.g. both cell-shape and ions, if cell-shape
and ionic relaxation are requested. Once convergence is achived, it locks the
cell-shape and relaxes the ions only. Finally, it performs a final static
calculation for maximum accuracy.

The functional is derived from :py:class:`~functional.Vasp`. In practice, this
means that whatever works for :py:class:`~functional.Vasp` works for
:py:class:`~relax.Relax`. However, it does accept a few extra attributes,
described below:

      first_trial
        A dictionary with parameters which are used only for the very first
        VASP calculation. It can be used to accelerate the first step of the
        relaxation if starting far from the optimum. For instance, it could be
        ``{'encut': 0.8}`` to first converge the structure with a smaller
        cutoff. Defaults to empty dictionary.

      maxcalls
        An interger which denotes the maximum number of calls to VASP before
        aborting. Defaults to 10.

      keepsteps
        If True, intermediate steps are kept. If False, intermediate steps are
        erased.

      relaxation
        Degrees of freedom to relax. It should be either "cellshape" or "ionic"
        or both. Same as for :py:class:`~functional.Vasp`.

      nofail
        If True, will not fail if convergence is not achieved. Just keeps going. 
        Defaults to False.

      convergence
        Convergence criteria. If ``minrelsteps`` is positive, it is only
        checked after ``minrelsteps`` have been performed. Convergence is
        checked according to last VASP run, not from one VASP run to another.
        Eg. If a positive real number, convergence is achieved when the
        difference between the last two total-energies of the current run fall
        below that real number (times structure size), not when the total
        energies of the last two runs fall below that number. Faster, but
        possibly less safe.

        * None: defaults to ``vasp.ediff * 1e1``
        * positive real number: energy convergence criteria in eV per atom. 
        * negative real number: force convergence criteria in eV/angstrom. 
        * callable: Takes an extraction object as input. Should return True if
          convergence is achieved and False otherwise.

      minrelsteps

        Fine tunes how convergence criteria is applied.
        
        * positive: at least ``minrelsteps`` calls to VASP are performed before
          checking for convergence. If ``relaxation`` contains "cellshape",
          then these calls occur during cellshape relaxation. If it does not,
          then the calls occur during the ionic relaxations. The calls do count
          towards ``maxcalls``.
        * negative (default): argument is ignored.


.. note::

   There is also a function :py:func:`relax.relax`, which does the exact same
   thing as the class. Whether to use one or the other is a matter of taste.
   And there's also a generator :py:func:`relax.iter_relax` over actual calls
   to the VASP_ binary.  Internally, it allows Pylada to launch different
   relaxations from different calculations side by side. In practice, both
   :py:func:`relax.relax` and :py:class:`relax.Relax` make calls to
   :py:func:`relax.iter_relax`.



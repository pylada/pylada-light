Relaxation methods
------------------

.. automodule:: pylada.vasp.relax
.. moduleauthor:: Mayeul d'Avezac <mayeul.davezac@nrel.gov>

.. currentmodule:: pylada.vasp.relax
.. autofunction:: relax

   Contains an :py:class:`Extract <RelaxExtract>` attribute which can be used
   to instantiate the relevant extraction object.

.. autofunction:: iter_relax

   Contains an :py:class:`Extract <RelaxExtract>` attribute which can be used
   to instantiate the relevant extraction object.

.. autoclass:: Relax
   :show-inheritance:

   .. attribute:: first_trial 

      Holds parameters which are used only for the very first VASP calculation.
      It can be used to accelerate the first step of the relaxation if starting
      far from the optimum. Defaults to empty dictionary.

   .. attribute:: maxcalls

      Maximum number of calls to VASP before aborting. Defaults to 10.

   .. attribute:: keepsteps

      If True, intermediate steps are kept. If False, intermediate steps are erased.

   .. attribute:: nofail

      If True, will not fail if convergence is not achieved. Just keeps going. Defaults to False.

   .. attribute:: convergence

        Convergence criteria. If ``minrelsteps`` is positive, it is only
        checked after ``minrelsteps`` have been performed.

        * None: defaults to ``vasp.ediff * 1e1``
        * positive real number: energy convergence criteria in eV. 
        * negative real number: force convergence criteria in eV/angstrom. 
        * callable: Takes an extraction object as input. Should return True if
          convergence is achieved and False otherwise.
          
   .. attribute:: minrelsteps

        Fine tunes how convergence criteria is applied.
        
        * positive: at least ``minrelsteps`` calls to VASP are performed before
          checking for convergence. If ``relaxation`` contains "cellshape",
          then these calls occur during cellshape relaxation. If it does not,
          then the calls occur during the ionic relaxations. The calls do count
          towards ``maxcalls``.
        * negative (default): argument is ignored.
      

   .. automethod:: __call__
   .. automethod:: iter

   .. attribute:: Extract

      Class :py:class:`RelaxExtract`. When called, it creates the appropriate
      relaxation object.

.. autofunction:: epitaxial
   
   Contains an :py:class:`Extract <RelaxExtract>` attribute which can be used
   to instantiate the relevant extraction object.


.. autofunction:: iter_epitaxial

   Contains an :py:class:`Extract <RelaxExtract>` attribute which can be used
   to instantiate the relevant extraction object.

.. autoclass:: Epitaxial
   :show-inheritance:

   .. automethod:: __call__
   .. automethod:: iter

   .. attribute:: direction

        Epitaxial direction. Defaults to [0, 0, 1].

   .. attribute:: epiconv

        Convergence criteria of the total energy.

   .. attribute:: Extract

      Class :py:class:`RelaxExtract`. When called, it creates the appropriate
      relaxation object.

.. autoclass:: RelaxExtract
   :show-inheritance:

   .. autoattribute:: details

      :py:class:`~pylada.vasp.extract.MassExtract` instance which maps extraction
      objects for intermediate steps in the 'relax_cellshape' and 'relax_ions'
      subdirectories.

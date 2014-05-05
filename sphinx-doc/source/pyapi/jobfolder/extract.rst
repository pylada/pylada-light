Extract base classes
********************

.. automodule:: pylada.jobfolder.extract
.. moduleauthor:: Mayeul d'Avezac

.. autoclass:: AbstractMassExtract
   :show-inheritance: AbstractMassExtract

   .. automethod:: __init__

   .. autoattribute:: rootpath
   .. autoattribute:: excludes

   .. attribute:: view

      Current view into the map of extraction object. Filters out the names
      which do not fit the view. It should be either a regex or a unix
      file-completion like string. The behavior is controled by
      :py:data:`pylada.unix_re`

   .. automethod:: __getitem__
   .. automethod:: __contains__

   .. automethod:: __iter__
   .. automethod:: iteritems
   .. automethod:: itervalues
   .. automethod:: iterkeys
   .. automethod:: items
   .. automethod:: values
   .. automethod:: keys

   .. automethod:: avoid
   .. automethod:: uncache
   .. automethod:: iterfiles

   .. automethod:: shallow_copy


.. seealso:: :py:class:`pylada.vasp.extract.MassExtract` extracts a directory and
             its subdirectories for vasp calculations.

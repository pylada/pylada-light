pylada.vasp.extract.mixin
*************************
.. automodule:: pylada.vasp.extract.mixin
.. autoclass:: pylada.vasp.extract.mixin.IOMixin
   :show-inheritance:
   :members:
   :inherited-members:
   
   .. automethod:: __outcar__()->file object
   .. automethod:: __contcar__()->file object

.. autoclass:: pylada.vasp.extract.mixin.OutcarSearchMixin
   :show-inheritance:
   :members:
   :inherited-members:

   .. automethod:: _search_OUTCAR(regex, flags=0)->re.match
   .. automethod:: _rsearch_OUTCAR(regex, flags=0)->re.match
   .. automethod:: _find_first_OUTCAR(regex, flags=0)->re.match
   .. automethod:: _find_last_OUTCAR(regex, flags=0)->re.match

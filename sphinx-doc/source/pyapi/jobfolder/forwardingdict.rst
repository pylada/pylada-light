Forwarding dictionary
*********************

.. automodule:: pylada.jobfolder.forwarding_dict
.. moduleauthor:: Mayeul d'Avezac
.. autoclass:: ForwardingDict
   :show-inheritance:
  
   .. automethod:: __init__

   .. attribute:: readonly

      Whether items can be modified in parallel using attribute syntax. 

   .. attribute:: naked_end
    
      Whether last item is returned as is or wrapped in ForwardingDict.

   .. attribute:: only_existing

      Whether attributes can be added or only modified.

   .. attribute:: dictionary
    
      The dictionary for which to unroll attributes. 

   .. autoattribute:: root
   .. autoattribute:: parent

   .. automethod:: keys

   .. automethod:: copy

   .. automethod:: __getitem__
   .. automethod:: __setitem__
   .. automethod:: __getattr__
   .. automethod:: __setattr__

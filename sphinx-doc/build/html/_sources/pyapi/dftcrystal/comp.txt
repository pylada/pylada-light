Computational and Hamiltonian parameters
----------------------------------------

.. currentmodule:: pylada.dftcrystal.electronic

.. autoclass:: Electronic
   :show-inheritance:
   :members:
   :member-order: groupwise
   :exclude-members: read_input, output_map, raw, dft

   .. automethod:: add_keyword

   .. attribute:: dft
     
      Attribute block which holds parameters defining the Hamiltonian used in
      the calculation. It is an instance of
      :py:class:`~pylada.dftcrystal.hamiltonian.Dft`.

.. currentmodule:: pylada.dftcrystal.hamiltonian

.. autoclass:: Dft
   :show-inheritance:
   :member-order: groupwise
   :exclude-members: read_input, output_map, add_keyword, raw
 
   .. automethod:: add_keyword

   .. attribute:: exchange 
   
      Name of the exchange functional to use.  It should be one of the
      following: 'becke', 'lda', 'pbe', 'pbesol', 'pwgga', 'sogga', 'vbh',
      'wcgga', or None(default).

   .. attribute:: correlat 
   
      Name of the correlation functional to use. It can be one of the
      following: 'lyp', 'p86', 'pbe', 'pbesol', 'pwgga', 'pwlsd', 'pz',
      'vbh', 'wl', 'vwn', or None. 
  
   .. attribute:: hybrid   
    
      Amount of exchange to add to functional. It should be a floating point or
      None.

   .. attribute:: nonlocal
   
      Non-local weights on exchange-correlation. A tuple of two floating point
      which sets the weights of the non-local part of the exchange (first) and
      of the correlations (second). It can also be None.

   .. attribute:: spin     
   
      If True, then perform spin-polarized calculation. It should be
      None(default), True, or False.
   
   .. attribute:: b3lyp 
   
      Sets :py:attr:`exchange`, :py:attr:`correlat`, :py:attr:`nonlocal`
      to the B3LIP functional. Since it acts on/checks upon other attributes,
      it can only be set to True.
  
   .. attribute b3pw 
   
      Sets :py:attr:`exchange`, :py:attr:`correlat`, :py:attr:`nonlocal`
      correctly for the B3PW functional. Since it acts on/checks upon other
      attributes, it can only be set to True. 

   .. attribute:: pbe0 
   
      Sets :py:attr:`exchange`, :py:attr:`correlat`, :py:attr:`nonlocal`
      correctly for the PBE0 functional. Since it acts on/checks upon other
      attributes, it can only be set to True.

   .. attribute:: soggaxc
   
      Sets :py:attr:`exchange`, :py:attr:`correlat`, :py:attr:`nonlocal`
      correctly for the SOGGAXC functional. Since it acts on/checks upon other
      attributes, it can only be set to True.
   
   .. attribute:: angular
   
      Contains two attributes ``intervals`` and ``levels`` which can be used
      to set the angular grid.

   .. attribute:: radial

      Contains two attributes ``intervals`` and ``nbpoints`` which can be used
      to set the radial integration grid.
   
   .. attribute:: lgrid 
   
      Preset large integration grid. 

   .. attribute:: xlgrid  
   
      Preset extra large integration grid. 
   
   .. attribute:: xxlgrid 
                         
      Preset extra extra large integration grid.

   .. attribute:: tollgrid 
   
      DFT grid weight tolerance. Should be None(default) or an integer.
  
   .. attribute:: tolldens 
   
      DFT density tolerance Should be None(default) or an integer.

Geometry optimization block
---------------------------
.. currentmodule:: pylada.dftcrystal.optgeom

.. autoclass:: OptGeom
   :show-inheritance:
   :members:
   :inherited-members:
   :exclude-members: fulloptg, cellonly, itatocell, interdun, add_keyword,
                     read_input, output_map, bhor

   .. method:: add_keyword(name, value=None)

      Creates a new parameter. The parameter will appear in the input within
      the ``OPTGEOM`` block. If ``value`` is ``None`` or True, then it will
      result in the following input:

        | OPTGEOM
        | NAME
        | END OPTGEOM
      
      The name is automatically in capital letters. Other keywords from the
      block were removed for clarity.
      If `value` is a string, then it is printed as is:

      >>> functional.optgeom.add_keyword('hello')
      >>> functional.optgeom.hello = "This\nis\na string"

      The above will create the formatted output:

        | OPTGEOM
        | HELLO
        | THIS
        | IS
        | A STRING
        | END OPTGEOM

      By formatting a string, any input, however complex, can be given.
      However, simple cases such as an integer, are handled sensibly:

      >>> functional.optgeom.hello = 2*5

        | OPTGEOM
        | HELLO
        | 10
        | END OPTGEOM

      Floating point numbers, and mixed lists of integers, floating points, and
      strings are also handled sensibly. 

      This function makes it easy to add new keywords to the ``OPTGEOM`` block.

   .. attribute:: fulloptg

     Optimization of all degrees of freedom: volume, cell-shape, ionic
     positions. 
     
     This is an instance of :py:class:`GeometryOpt`. It excludes
     :py:attr:`cellonly`, :py:attr:`itatocell`, :py:attr:`interdun`.

     >>> functional.optgeom.fulloptg = True
     >>> functional.optgeom.cellonly, functional.optgeom.itatocell, functional.optgeom.interdun
     False, False, False

     It can also be made to optimize at constant volume if :py:attr:`cvolopt` is True.

   .. attribute:: cellonly

     Optimization of cell-shape at constant atomic-position.
     
     This is an instance of :py:class:`GeometryOpt`. It excludes
     :py:attr:`fulloptg`, :py:attr:`itatocell`, :py:attr:`interdun`.

     >>> functional.optgeom.cellonly = True
     >>> functional.optgeom.fulloptg, functional.optgeom.itatocell, functional.optgeom.interdun
     False, False, False

     It can also be made to optimize at constant volume if :py:attr:`cvolopt` is True.

   .. attribute:: itatocell

     Iterative optimization of cell-shape <--> atomic positions. 
     
     This is an instance of :py:class:`GeometryOpt`. It excludes
     :py:attr:`cellonly`, :py:attr:`fulloptg`, :py:attr:`interdun`.

     >>> functional.optgeom.itatocell = True
     >>> functional.optgeom.fulloptg, functional.optgeom.cellonly, functional.optgeom.interdun
     False, False, False

   .. attribute:: interdun

     If True, turns on constrained optimization. See CRYSTAL_ manual.
     
     This is an instance of :py:class:`GeometryOpt`. It excludes
     :py:attr:`cellonly`, :py:attr:`fulloptg`, :py:attr:`interdun`.

     >>> functional.optgeom.itatocell = True
     >>> functional.optgeom.fulloptg, functional.optgeom.cellonly, functional.optgeom.interdun
     False, False, False

   .. attribute:: cvolopt
      
      If True *and* if one of :py:attr:`fulloptg` or :py:attr:`cellonly` is
      True, then performs constant volume optimization.
  
   .. attribute:: bhor
  
      Bhor radius, defined as in CRYSTAL_, 0.5291772083 angstrom.

.. autoclass:: ExclAttrBlock
   :show-inheritance:
   :members:
   :exclude-members: keyword

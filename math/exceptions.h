/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab
  
   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
   large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
   crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
   is able to organise and launch computational jobs on PBS and SLURM.
  
   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
   Public License as published by the Free Software Foundation, either version 3 of the License, or (at
   your option) any later version.
  
   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.
  
   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

#if PYLADA_MATH_MODULE != 1
  namespace Pylada
  {
    namespace error
    {
      //! Root of math errors.
      struct math : virtual root {};
      //! Thrown two arrays have different sizes.
      struct array_of_different_sizes: virtual math {};
      //! \brief Thrown when an structure is not the expected supercell of a lattice.
      //! \details This error may occur in methods which expects structures to be
      //!          supercells of a lattice, without allowance for relaxation. 
      struct unideal_lattice: virtual math, virtual root {};
      //! Thrown when an atomic position is unexpectedly off-lattice.
      struct off_lattice_position: virtual math, virtual unideal_lattice {};
      //! Thrown when a structure is not a supercell of an ideal lattice.
      struct not_a_supercell: virtual math, virtual unideal_lattice {};
      //! Thrown when a matrix is singular.
      struct singular_matrix: virtual math, virtual input {};
      //! Thrown when a matrix has a negative determinant.
      struct negative_volume: virtual math, virtual input {};
    }
  }
# endif
  
  
  
  
  
  
  
  
  
  
  
  

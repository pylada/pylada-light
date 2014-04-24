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

# ifdef PYLADA_INTEGRAL
#   error PYLADA_INTEGRAL already defined.
# endif
# ifdef PYLADA_REAL
#   error PYLADA_REAL already defined.
# endif
# ifdef PYLADA_ARITH
#   error PYLADA_ARITH already defined.
# endif
# define PYLADA_INTEGRAL typename boost::enable_if<boost::is_integral<T_ARG>, bool> :: type
# define PYLADA_REAL typename boost::enable_if<boost::is_floating_point<T_ARG>, bool> :: type
# define PYLADA_ARITH typename boost::enable_if<boost::is_arithmetic<T_ARG>, bool> :: type
  //! \brief True if \f$|\_a - \_b| < \_tol\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_REAL eq(T_ARG const& _a, T_ARG const& _b, T_ARG const _tol = types::tolerance )
    { return std::abs(_a-_b) < _tol; }
  //! \brief True if \f$|\_a - \_b| < \_tol\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL eq(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)
    { return std::abs(_a-_b) < _tol; }
  //! \brief True if \f$\_a == \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL eq(T_ARG const& _a, T_ARG const& _b) { return _a == _b; }

  //! \brief True if \f$|\_a - \_b| < \_tol\f$ or \f$\_a < \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_REAL leq(T_ARG const& _a, T_ARG const& _b, T_ARG const _tol = types::tolerance )
    { return _a < _b + _tol; }
  //! \brief True if \f$|\_a - \_b| < \_tol\f$ or \f$\_a < \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL leq(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)
    { return _a <= _b + _tol; }
  //! \brief True if \f$\_a <= \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL leq(T_ARG const& _a, T_ARG const& _b) { return _a <= _b; }

  //! \brief True if \f$|\_a - \_b| < \_tol\f$ or \f$\_a > \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_REAL geq(T_ARG const& _a, T_ARG const& _b, T_ARG const _tol = types::tolerance )
    { return _a > _b - _tol; }
  //! \brief True if \f$|\_a - \_b| < \_tol\f$ or \f$\_a > \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL geq(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)
    { return _a >= _b - _tol; }
  //! \brief True if \f$\_a >= \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_INTEGRAL geq(T_ARG const& _a, T_ARG const& _b) { return _a >= _b; }

  //! \brief True if \f$|\_a - \_b| > \_tol\f$ and \f$\_a < \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH lt(T_ARG const& _a, T_ARG const& _b)   { return not geq(_a, _b); }
  //! \brief True if \f$|\_a - \_b| > \_tol\f$ and \f$\_a < \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH lt(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)   { return not geq(_a, _b, _tol); }

  //! \brief True if \f$|\_a - \_b| > \_tol\f$ and \f$\_a > \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH gt(T_ARG const& _a, T_ARG const& _b)   { return not leq(_a, _b); }
  //! \brief True if \f$|\_a - \_b| > \_tol\f$ and \f$\_a > \_b|\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH gt(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)   { return not leq(_a, _b, _tol); }

  //! \brief True if \f$|\_a - \_b| > \_tol\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH neq(T_ARG const& _a, T_ARG const& _b)   { return not eq(_a, _b); }
  //! \brief True if \f$|\_a - \_b| > \_tol\f$.
  //! \details _tol should be positive. This function implements fuzzy math
  //!               across most numeric types.
  template< class T_ARG>
  inline PYLADA_ARITH neq(T_ARG const& _a, T_ARG const& _b, T_ARG const& _tol)   { return not eq(_a, _b, _tol); }
  
  //! True if the number is an integer.
  template<class T_ARG>
  inline PYLADA_REAL is_integer(T_ARG const& x, T_ARG const _tol = types::tolerance)
      { return eq(x, std::floor(x+0.1), _tol); }
  //! True if the number is an integer.
  template<class T_ARG> inline PYLADA_INTEGRAL is_integer(T_ARG const&, T_ARG const&) { return true; }
  //! True if the number is an integer.
  template<class T_ARG> inline PYLADA_INTEGRAL is_integer(T_ARG const&) { return true; }

  //! \brief returns true if \a _a  == 0.
  //! \details if \a T_ARG is a types::real, return true if 
  //!          \a _a < types::tolerance.
  template<class T_ARG> inline PYLADA_ARITH is_null(T_ARG const& _a, T_ARG const& _tol)
    { return eq(_a, T_ARG(0), _tol); }
  //! \brief returns true if \a _a  == 0.
  //! \details if \a T_ARG is a types::real, return true if 
  //!          \a _a < types::tolerance.
  template<class T_ARG> inline PYLADA_ARITH is_null(T_ARG const& _a) { return eq(_a, T_ARG(0)); }

  //! \brief returns true if \a _a  == 0.
  //! \details if \a T_ARG is a types::real, return true if 
  //!          \a _a < types::tolerance.
  template<class T_ARG> inline PYLADA_ARITH is_identity(T_ARG const& _a, T_ARG const& _tol)
    { return eq(_a, T_ARG(1), _tol); }
  //! \brief returns true if \a _a  == 0.
  //! \details if \a T_ARG is a types::real, return true if 
  //!          \a _a < types::tolerance.
  template<class T_ARG> inline PYLADA_ARITH is_identity(T_ARG const& _a) { return eq(_a, T_ARG(1)); }

  //! Casts to lower integer using boost::numeric.
  template<class T_ARG>
    inline typename boost::enable_if<boost::is_integral<T_ARG>, T_ARG> :: type 
      floor_int(T_ARG const &_a) { return _a; }
  //! Casts to lower integer using boost::numeric.
  template<class T_ARG> 
    inline typename boost::enable_if<boost::is_floating_point<T_ARG>, types::t_int> :: type 
      floor_int(T_ARG const &_a)
      { 
        typedef boost::numeric::converter<types::t_int,types::t_real> converter;
        return converter::nearbyint( std::floor(_a + types::roundoff) );
      }
# undef PYLADA_INTEGRAL
# undef PYLADA_REAL
# undef PYLADA_ARITH

#endif

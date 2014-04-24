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

#if PYLADA_PYTHON_MODULE != 1
  //! An mpl integer defining the type.
  template<class T> class type;
  
# ifdef NUMPY_HAS_LONG_DOUBLE
    //! numpy identifier for long doubles.
    template<> struct type<npy_longdouble> : public boost::mpl::int_<NPY_LONGDOUBLE> 
    {
      //! Original type.
      typedef npy_longdouble np_type;
    };
# endif
  //! numpy identifier for doubles.
  template<> struct type<npy_double> : public boost::mpl::int_<NPY_DOUBLE> 
  {
    //! Original type.
    typedef npy_double np_type;
  };
  //! numpy identifier for float.
  template<> struct type<npy_float> : public boost::mpl::int_<NPY_FLOAT> 
  {
    //! Original type.
    typedef npy_float np_type;
  };
  //! numpy identifier for long long.
  template<> struct type<npy_longlong> : public boost::mpl::int_<NPY_LONGLONG> 
  {
    //! Original type.
    typedef npy_longlong np_type;
  };
  //! numpy identifier for unsigned long.
  template<> struct type<npy_ulonglong> : public boost::mpl::int_<NPY_ULONGLONG> 
  {
    //! Original type.
    typedef npy_ulonglong np_type;
  };
  //! numpy identifier for long.
  template<> struct type<npy_long> : public boost::mpl::int_<NPY_LONG> 
  {
    //! Original type.
    typedef npy_long np_type;
  };
  //! numpy identifier for unsigned long.
  template<> struct type<npy_ulong> : public boost::mpl::int_<NPY_ULONG> 
  {
    //! Original type.
    typedef npy_ulong np_type;
  };
  //! numpy identifier for int.
  template<> struct type<npy_int> : public boost::mpl::int_<NPY_INT> 
  {
    //! Original type.
    typedef npy_int np_type;
  };
  //! numpy identifier for unsigned int.
  template<> struct type<npy_uint> : public boost::mpl::int_<NPY_UINT> 
  {
    //! Original type.
    typedef npy_uint np_type;
  };
  //! numpy identifier for short.
  template<> struct type<npy_short> : public boost::mpl::int_<NPY_SHORT> 
  {
    //! Original type.
    typedef npy_short np_type;
  };
  //! numpy identifier for unsigned short.
  template<> struct type<npy_ushort> : public boost::mpl::int_<NPY_USHORT> 
  {
    //! Original type.
    typedef npy_ushort np_type;
  };
  //! numpy identifier for byte.
  template<> struct type<npy_byte> : public boost::mpl::int_<NPY_BYTE> 
  {
    //! Original type.
    typedef npy_byte np_type;
  };
  //! numpy identifier for unsigned byte..
  template<> struct type<npy_ubyte> : public boost::mpl::int_<NPY_UBYTE> 
  {
    //! Original type.
    typedef npy_ubyte np_type;
  };
# ifdef NUMPY_HAS_BOOL
    //! numpy identifier for bool.
    template<> struct type<npy_bool> : public boost::mpl::int_<NPY_BOOL> 
    {
      //! Original type.
      typedef npy_bool np_type;
    };
# else
    //! numpy identifier for bool.
    template<> struct type<bool> : public boost::mpl::int_<NPY_BOOL> 
    {
      //! Original type.
      typedef npy_bool np_type;
    };
# endif 
  
  //! Returns true if object is float.
  inline bool is_float(PyObject *_obj_ptr)
  {
    int const nptype = PyArray_ObjectType(_obj_ptr, 0);
    return    nptype ==  NPY_FLOAT
           or nptype ==  NPY_DOUBLE
           or nptype ==  NPY_LONGDOUBLE;
  }
  
  //! Returns true if object is complex.
  inline bool is_complex(PyObject *_obj_ptr)
  {
    int const nptype = PyArray_ObjectType(_obj_ptr, 0);
    return    nptype ==  NPY_CDOUBLE; 
  };
  //! Returns true if object is integer.
  inline bool is_integer(PyObject *_obj_ptr)
  {
    int const nptype = PyArray_ObjectType(_obj_ptr, 0);
    return    nptype ==  NPY_INT
           or nptype ==  NPY_UINT
           or nptype ==  NPY_LONG
           or nptype ==  NPY_ULONG
           or nptype ==  NPY_LONGLONG
           or nptype ==  NPY_ULONGLONG
           or nptype ==  NPY_BYTE
           or nptype ==  NPY_UBYTE
           or nptype ==  NPY_SHORT
           or nptype ==  NPY_USHORT;
  }
  //! Returns true if object is boolean.
  inline bool is_bool(PyObject *_obj_ptr) { return PyArray_ObjectType(_obj_ptr, 0) ==  NPY_BOOL; }
  
  //! Returns true if downcasting from PyObject to T. 
  template<class T> bool is_downcasting(PyObject *_obj_ptr) 
  {
    switch(PyArray_ObjectType(_obj_ptr, 0))
    {
      case NPY_FLOAT     : return sizeof(type<npy_float>::np_type) > sizeof(T);
      case NPY_DOUBLE    : return sizeof(type<npy_double>::np_type) > sizeof(T);
      //case NPY_LONGDOUBLE: return sizeof(type<npy_longdouble>::np_type) > sizeof(T);
      case NPY_INT       : return sizeof(type<npy_int>::np_type) > sizeof(T);
      case NPY_UINT      : return sizeof(type<npy_uint>::np_type) > sizeof(T);
      case NPY_LONG      : return sizeof(type<npy_long>::np_type) > sizeof(T);
      case NPY_ULONG     : return sizeof(type<npy_ulong>::np_type) > sizeof(T);
      case NPY_LONGLONG  : return sizeof(type<npy_longlong>::np_type) > sizeof(T);
      case NPY_ULONGLONG : return sizeof(type<npy_ulonglong>::np_type) > sizeof(T);
      case NPY_BYTE      : return sizeof(type<npy_byte>::np_type) > sizeof(T);
      case NPY_UBYTE     : return sizeof(type<npy_ubyte>::np_type) > sizeof(T);
      case NPY_SHORT     : return sizeof(type<npy_short>::np_type) > sizeof(T);
      case NPY_USHORT    : return sizeof(type<npy_ushort>::np_type) > sizeof(T);
      default: break;
    };
    PYLADA_PYERROR(ValueError, "Unknown numpy array type.");
  }

  //! Casts data to requested type
  template<class T> T cast_data(void const * const _data, int const _type) 
  {
    switch(_type)
    {
      case NPY_FLOAT     : return (T)*((type<npy_float>::np_type const * const)      _data);
      case NPY_DOUBLE    : return (T)*((type<npy_double>::np_type const * const)     _data);
      //case NPY_LONGDOUBLE: return (T)*((type<npy_longdouble>::np_type const * const) _data);
      case NPY_INT       : return (T)*((type<npy_int>::np_type const * const)        _data);
      case NPY_UINT      : return (T)*((type<npy_uint>::np_type const * const)       _data);
      case NPY_LONG      : return (T)*((type<npy_long>::np_type const * const)       _data);
      case NPY_ULONG     : return (T)*((type<npy_ulong>::np_type const * const)      _data);
      case NPY_LONGLONG  : return (T)*((type<npy_longlong>::np_type const * const)   _data);
      case NPY_ULONGLONG : return (T)*((type<npy_ulonglong>::np_type const * const)  _data);
      case NPY_BYTE      : return (T)*((type<npy_byte>::np_type const * const)       _data);
      case NPY_UBYTE     : return (T)*((type<npy_ubyte>::np_type const * const)      _data);
      case NPY_SHORT     : return (T)*((type<npy_short>::np_type const * const)      _data);
      case NPY_USHORT    : return (T)*((type<npy_ushort>::np_type const * const)     _data);
      default: break;
    };
    PYLADA_PYERROR(ValueError, "Unknown numpy array type.");
    return T(0);
  }
  

  //! Converts data from numpy iterator to requested type.
  template<class T>
    T to_type(PyObject* _iter, int const _type, int const _offset = 0)
    {
      char const * const ptr_data = ((char const *) PyArray_ITER_DATA(_iter) + _offset);
      switch(_type)
      {
        case NPY_FLOAT     :
          return static_cast<T>(*((type<npy_float>::np_type*)      ptr_data));
        case NPY_DOUBLE    :
          return static_cast<T>(*((type<npy_double>::np_type*)     ptr_data));
        //case NPY_LONGDOUBLE:
        //  return static_cast<T>(*((type<npy_longdouble>::np_type*) ptr_data));
        case NPY_INT       :
          return static_cast<T>(*((type<npy_int>::np_type*)        ptr_data));
        case NPY_UINT      :
          return static_cast<T>(*((type<npy_uint>::np_type*)       ptr_data));
        case NPY_LONG      :
          return static_cast<T>(*((type<npy_long>::np_type*)       ptr_data));
        case NPY_ULONG     :
          return static_cast<T>(*((type<npy_ulong>::np_type*)      ptr_data));
        case NPY_LONGLONG  :
          return static_cast<T>(*((type<npy_longlong>::np_type*)   ptr_data));
        case NPY_ULONGLONG :
          return static_cast<T>(*((type<npy_ulonglong>::np_type*)  ptr_data));
        case NPY_BYTE      :
          return static_cast<T>(*((type<npy_byte>::np_type*)       ptr_data));
        case NPY_UBYTE     :
          return static_cast<T>(*((type<npy_ubyte>::np_type*)      ptr_data));
        case NPY_SHORT     :
          return static_cast<T>(*((type<npy_short>::np_type*)      ptr_data));
        case NPY_USHORT    :
          return static_cast<T>(*((type<npy_ushort>::np_type*)     ptr_data));
        default: break;
      };
      PYLADA_PYERROR(ValueError, "Unknown numpy type on input.");
      return T(-1);
    }
#endif


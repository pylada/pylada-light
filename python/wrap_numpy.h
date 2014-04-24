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
  //! Convert/wrap a matrix to numpy.
  template<class T_DERIVED>
    PyObject* wrap_to_numpy(Eigen::DenseBase<T_DERIVED> const &_in, PyObject *_parent = NULL)
    {
      npy_intp dims[2] = { _in.rows(), _in.cols() };
      typedef numpy::type<typename Eigen::DenseBase<T_DERIVED>::Scalar> t_ScalarType;
      PyArrayObject *result = _parent == NULL ?
        (PyArrayObject*) PyArray_ZEROS(_in.cols() > 1? 2: 1, dims, t_ScalarType::value, _in.IsRowMajor?0:1):
        (PyArrayObject*) PyArray_SimpleNewFromData(_in.cols() > 1? 2: 1, dims, t_ScalarType::value,
                                   (void*)(&_in(0,0)));
      if(result == NULL) return NULL;
      // If has a parent, do not copy data, just incref it as base.
      if(_parent != NULL) 
      {
        // For some reason, eigen is column major, whereas c++ is generally row major.
        if(not _in.IsRowMajor) 
          PyArray_CLEARFLAGS(result, NPY_ARRAY_C_CONTIGUOUS);
        else if( _in.IsRowMajor) 
          PyArray_ENABLEFLAGS(result, NPY_ARRAY_C_CONTIGUOUS);
        if(_in.cols() == 1)
          PyArray_STRIDES(result)[0] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
        else if(_in.IsRowMajor) 
        {
          PyArray_STRIDES(result)[0] = _in.outerStride() * sizeof(typename t_ScalarType::np_type);
          PyArray_STRIDES(result)[1] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
        }
        else 
        {
          PyArray_STRIDES(result)[0] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
          PyArray_STRIDES(result)[1] = _in.outerStride() * sizeof(typename t_ScalarType::np_type);
        }
        Py_INCREF(_parent);
        PyArray_SetBaseObject(result, _parent);
      }
      // otherwise, copy data.
      else
      {
        for(int i(0); i < _in.rows(); ++i)
          for(int j(0); j < _in.cols(); ++j)
            *((typename t_ScalarType::np_type*) PyArray_GETPTR2(result, i, j)) = _in(i, j);
      }
      PyArray_CLEARFLAGS(result, NPY_ARRAY_WRITEABLE);
      return (PyObject*)result;
    }
  //! Convert/wrap a matrix to numpy.
  template<class T_DERIVED>
    PyObject* wrap_to_numpy(Eigen::DenseBase<T_DERIVED> &_in, PyObject *_parent = NULL)
    {
      npy_intp dims[2] = { _in.rows(), _in.cols() };
      typedef numpy::type<typename Eigen::DenseBase<T_DERIVED>::Scalar> t_ScalarType;
      PyArrayObject *result = _parent == NULL ?
        (PyArrayObject*) PyArray_ZEROS(_in.cols() > 1? 2: 1, dims, t_ScalarType::value, _in.IsRowMajor?0:1):
        (PyArrayObject*) PyArray_SimpleNewFromData(_in.cols() > 1? 2: 1, dims, t_ScalarType::value, &_in(0,0));
      if(result == NULL) return NULL;
      // If has a parent, do not copy data, just incref it as base.
      if(_parent != NULL) 
      {
        // For some reason, eigen is column major, whereas c++ is generally row major.
        if(PyArray_FLAGS(result) & NPY_ARRAY_C_CONTIGUOUS and not _in.IsRowMajor) 
          PyArray_CLEARFLAGS(result, NPY_ARRAY_C_CONTIGUOUS);
        else if( _in.IsRowMajor) 
          PyArray_ENABLEFLAGS(result, NPY_ARRAY_C_CONTIGUOUS);
        if(_in.cols() == 1)
          PyArray_STRIDES(result)[0] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
        else if(_in.IsRowMajor) 
        {
          PyArray_STRIDES(result)[0] = _in.outerStride() * sizeof(typename t_ScalarType::np_type);
          PyArray_STRIDES(result)[1] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
        }
        else 
        {
          PyArray_STRIDES(result)[0] = _in.innerStride() * sizeof(typename t_ScalarType::np_type);
          PyArray_STRIDES(result)[1] = _in.outerStride() * sizeof(typename t_ScalarType::np_type);
        }
        Py_INCREF(_parent);
        PyArray_SetBaseObject(result, _parent);
      }
      // otherwise, copy data.
      else
      {
        for(int i(0); i < _in.rows(); ++i)
          for(int j(0); j < _in.cols(); ++j)
            *((typename t_ScalarType::np_type*) PyArray_GETPTR2(result, i, j)) = _in(i, j);
      }
      PyArray_ENABLEFLAGS(result, NPY_ARRAY_WRITEABLE);
      return (PyObject*)result;
    }
  //! Converts an input sequence to a cell.
  template<class T_DERIVED>
    bool convert_to_matrix(PyObject *_in, Eigen::DenseBase<T_DERIVED> &_out)
    {
      Py_ssize_t const N0(_out.rows());
      Py_ssize_t const N1(_out.cols());
      if(PyArray_Check(_in))
      {
        PyArrayObject * const in_((PyArrayObject*)_in);
        if(PyArray_NDIM(in_) != 2)
        {
          npy_intp const n(PyArray_NDIM(in_));
          PYLADA_PYERROR_FORMAT(TypeError, "Expected a 2d array, got %id", int(n));
          return false;
        }
        if(PyArray_DIM(in_, 0) != N0 or PyArray_DIM(in_, 1) != N1)
        {
          npy_intp const n0(PyArray_DIM(in_, 0));
          npy_intp const n1(PyArray_DIM(in_, 1));
          {                                                                         
            PyObject* err_module = PyImport_ImportModule("pylada.error");             
            if(err_module)                                                          
            {                                                                       
              PyObject *err_result = PyObject_GetAttrString(err_module, "TypeError");
              if(not err_result) Py_DECREF(err_module);                             
              else                                                                  
              {                                                                     
                PyErr_Format(err_result, "Expected a %ix%i array, got %ix%i", int(N0), int(N1), int(n0), int(n1));                           
                Py_DECREF(err_module);                                              
                Py_DECREF(err_result);                                              
              }                                                                     
            }                                                                       
          }
          return false;
        }
        python::Object iterator = PyArray_IterNew(_in);
        if(not iterator) return false;
        int const type = PyArray_TYPE(in_);
#       ifdef  PYLADA_NPYITER
#         error PYLADA_NPYITER already defined
#       endif
#       define PYLADA_NPYITER(TYPE, NUM_TYPE)                                             \
          if(type == NUM_TYPE)                                                          \
          {                                                                             \
            for(Py_ssize_t i(0); i < N0*N1; ++i)                                        \
            {                                                                           \
              if(not PyArray_ITER_NOTDONE(iterator.borrowed()))                         \
              {                                                                         \
                PYLADA_PYERROR(TypeError, "Numpy array too small.");                      \
                return false;                                                           \
              }                                                                         \
              _out(i/N1, i%N1) = *((TYPE*) PyArray_ITER_DATA(iterator.borrowed()));     \
              PyArray_ITER_NEXT(iterator.borrowed());                                   \
            }                                                                           \
            if(PyArray_ITER_NOTDONE(iterator.borrowed()))                               \
            {                                                                           \
              PYLADA_PYERROR(TypeError, "Numpy array too long.");                         \
              return false;                                                             \
            }                                                                           \
          }
        PYLADA_NPYITER( npy_float,      NPY_FLOAT)      
        else PYLADA_NPYITER( npy_double,     NPY_DOUBLE     )
        else PYLADA_NPYITER( npy_longdouble, NPY_LONGDOUBLE )
        else PYLADA_NPYITER( npy_int,        NPY_INT        )
        else PYLADA_NPYITER( npy_uint,       NPY_UINT       )
        else PYLADA_NPYITER( npy_long,       NPY_LONG       )
        else PYLADA_NPYITER( npy_longlong,   NPY_LONGLONG   )
        else PYLADA_NPYITER( npy_ulonglong,  NPY_ULONGLONG  )
        else PYLADA_NPYITER( npy_ubyte,      NPY_BYTE       )
        else PYLADA_NPYITER( npy_short,      NPY_SHORT      )
        else PYLADA_NPYITER( npy_ushort,     NPY_USHORT     )
        else
        {
          PYLADA_PYERROR(TypeError, "Unknown numpy array type.");
          return false;
        }
#       undef PYLADA_NPYITER
      } // numpy array
      else 
      {
        python::Object i_outer = PyObject_GetIter(_in);
        if(not i_outer)
        { 
          if(not PyErr_Occurred()) 
            PYLADA_PYERROR(TypeError, "Argument cannot be converted to matrix.");
           return false; 
        }
        python::Object outer(PyIter_Next(i_outer.borrowed()));
        if(not outer.is_valid())
        { 
          if(not PyErr_Occurred()) 
            PYLADA_PYERROR(TypeError, "Argument cannot be converted to matrix.");
          return false; 
        }
        if(not outer.hasattr("__iter__")) // except 9 in a row
        {
          Py_ssize_t i(0);
          for( ; outer.is_valid() and i < N0*N1;
               outer.reset(PyIter_Next(i_outer.borrowed())), ++i ) 
          {
            if(PyInt_Check(outer.borrowed())) _out(i/N0, i%N0) = PyInt_AS_LONG(outer.borrowed());
            else if(PyFloat_Check(outer.borrowed())) _out(i/N0, i%N0) = PyFloat_AS_DOUBLE(outer.borrowed());
            else
            { 
              PYLADA_PYERROR(TypeError, "Object should contains numbers only.");
              return false;
            }
          }
          if(outer.is_valid() or i != 9)
          {
            PYLADA_PYERROR(TypeError, "Expected 9 (NxN) numbers in input.");
            return false;
          }
        }    // N0*N1 in a row.
        else // expect N0 by N1
        {
          Py_ssize_t i(0);
          for( ; outer.is_valid() and i < N0;
               outer.reset(PyIter_Next(i_outer.borrowed())), ++i ) 
          {
            python::Object i_inner = PyObject_GetIter(outer.borrowed());
            if(not i_inner) return false;
            python::Object inner(PyIter_Next(i_inner.borrowed()));
            if(not inner) return false;
            Py_ssize_t j(0);
            for( ; inner.is_valid() and j < N1;
                 inner.reset(PyIter_Next(i_inner.borrowed())), ++j ) 
            {
              if(PyInt_Check(inner.borrowed())) _out(i, j) = PyInt_AS_LONG(inner.borrowed());
              else if(PyFloat_Check(inner.borrowed())) _out(i, j) = PyFloat_AS_DOUBLE(inner.borrowed());
              else
              { 
                PYLADA_PYERROR(TypeError, "Object should contains numbers only.");
                return false;
              }
            } // inner loop.
            if(inner.is_valid() or j != N1)
            {
              PYLADA_PYERROR(TypeError, "Not a NxN matrix of numbers.");
              return false;
            }
          } // outer loop.
          if(outer.is_valid() or i != N1)
          {
            PYLADA_PYERROR(TypeError, "Not a NxN matrix of numbers.");
            return false;
          }
        }
      } // sequence.
      return true;
    }
  //! Converts an input sequence to a cell.
  template<class T_DERIVED> 
    bool convert_to_vector(PyObject *_in, Eigen::DenseBase<T_DERIVED> &_out)
    {
      if(PyArray_Check(_in))
      {
        python::Object iterator = PyArray_IterNew(_in);
        if(not iterator) return false;
        int const type = PyArray_TYPE((PyArrayObject*)_in);
#       ifdef PYLADA_NPYITER
#         error PYLADA_NPYITER is already defined.
#       endif
#       define PYLADA_NPYITER(TYPE, NUM_TYPE)                                        \
          if(type == NUM_TYPE)                                                     \
          {                                                                        \
            for(size_t i(0); i < 3; ++i)                                           \
            {                                                                      \
              if(not PyArray_ITER_NOTDONE(iterator.borrowed()))                    \
              {                                                                    \
                PYLADA_PYERROR(TypeError, "Numpy array too small.");                 \
                return false;                                                      \
              }                                                                    \
              _out[i] = *((TYPE*) PyArray_ITER_DATA(iterator.borrowed()));         \
              PyArray_ITER_NEXT(iterator.borrowed());                              \
            }                                                                      \
            if(PyArray_ITER_NOTDONE(iterator.borrowed()))                          \
            {                                                                      \
              PYLADA_PYERROR(TypeError, "Numpy array too long.");                    \
              return false;                                                        \
            }                                                                      \
          }
        PYLADA_NPYITER( npy_float,      NPY_FLOAT)      
        else PYLADA_NPYITER( npy_double,     NPY_DOUBLE     )
        else PYLADA_NPYITER( npy_longdouble, NPY_LONGDOUBLE )
        else PYLADA_NPYITER( npy_int,        NPY_INT        )
        else PYLADA_NPYITER( npy_uint,       NPY_UINT       )
        else PYLADA_NPYITER( npy_long,       NPY_LONG       )
        else PYLADA_NPYITER( npy_longlong,   NPY_LONGLONG   )
        else PYLADA_NPYITER( npy_ulonglong,  NPY_ULONGLONG  )
        else PYLADA_NPYITER( npy_ubyte,      NPY_BYTE       )
        else PYLADA_NPYITER( npy_short,      NPY_SHORT      )
        else PYLADA_NPYITER( npy_ushort,     NPY_USHORT     )
        else
        {
          PYLADA_PYERROR(TypeError, "Unknown numpy array type.");
          return false;
        }
    #   undef PYLADA_NPYITER
      }
      else if(PyInt_Check(_in)) _out = Eigen::DenseBase<T_DERIVED>::Ones() * PyInt_AS_LONG(_in); 
      else if( boost::is_floating_point< typename Eigen::DenseBase<T_DERIVED> >::value 
               and PyFloat_Check(_in))
        _out = Eigen::DenseBase<T_DERIVED>::Ones() * PyFloat_AS_DOUBLE(_in); 
      else
      {
        python::Object i_outer = PyObject_GetIter(_in);
        if(not i_outer) { return false; }
        python::Object item(PyIter_Next(i_outer.borrowed()));
        size_t i(0);
        for( ; item.is_valid() and i < 3;
             item.reset(PyIter_Next(i_outer.borrowed())), ++i ) 
          if(PyInt_Check(item.borrowed()) == 1) _out[i] = PyInt_AS_LONG(item.borrowed());
          else if(PyFloat_Check(item.borrowed()) == 1) _out[i] = PyFloat_AS_DOUBLE(item.borrowed());
          else
          { 
            PYLADA_PYERROR(TypeError, "Input vector should contains numbers only.");
            return false;
          }
        if(item.is_valid())
        { 
          PYLADA_PYERROR(TypeError, "Input vector is too large.");
          return false;
        }
        else if (i != 3) 
        {
          PYLADA_PYERROR(TypeError, "Input vector is too small.");
          return false;
        }
      } 
      return true;
    }
#endif

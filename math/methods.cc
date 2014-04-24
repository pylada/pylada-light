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

PyObject* pyis_integer(PyObject *_module, PyObject* _in)
{
  using namespace Pylada;
  if(not PyArray_Check(_in))
  {
    PYLADA_PYERROR(TypeError, "Argument should be a numpy array.");
    return NULL;
  }
  int const type = PyArray_TYPE((PyArrayObject*)_in);
# ifdef  PYLADA_NPYITER
#   error PYLADA_NPYITER already defined
# endif
# define PYLADA_NPYITER(TYPE, NUM_TYPE)                                     \
    if(type == NUM_TYPE)                                                  \
    {                                                                     \
      python::Object iterator = PyArray_IterNew(_in);                     \
      while(PyArray_ITER_NOTDONE(iterator.borrowed()))                    \
      {                                                                   \
        TYPE const x = *((TYPE*)PyArray_ITER_DATA(iterator.borrowed()));  \
        if(not Pylada::math::eq(x, TYPE(std::floor(x+0.1))) )               \
          { Py_RETURN_FALSE; }                                            \
        PyArray_ITER_NEXT(iterator.borrowed());                           \
      }                                                                   \
      Py_RETURN_TRUE;                                                     \
    }
  PYLADA_NPYITER( npy_float,      NPY_FLOAT)      
  else PYLADA_NPYITER( npy_double,     NPY_DOUBLE     )
  else PYLADA_NPYITER( npy_longdouble, NPY_LONGDOUBLE )
  else if(    type == NPY_INT
           or type == NPY_UINT        
           or type == NPY_LONG        
           or type == NPY_LONGLONG    
           or type == NPY_ULONGLONG   
           or type == NPY_BYTE        
           or type == NPY_SHORT       
           or type == NPY_USHORT     ) Py_RETURN_TRUE;
  else
  {
    PYLADA_PYERROR(TypeError, "Unknown numpy array type.");
    return NULL;
  }
# undef PYLADA_NPYITER
}

PyObject* pyfloor_int(PyObject *_module, PyObject* _in)
{
  using namespace Pylada;
  if(not PyArray_Check(_in))
  {
    PYLADA_PYERROR(TypeError, "Argument should be a numpy array.");
    return NULL;
  }
  python::Object result = PyArray_SimpleNew( PyArray_NDIM((PyArrayObject*)_in),
                                             PyArray_DIMS((PyArrayObject*)_in), 
                                             NPY_LONG );
  if(not result) return NULL;
  python::Object iter_in = PyArray_IterNew(_in);
  if(not iter_in) return NULL;
  python::Object iter_out = PyArray_IterNew(result.borrowed());
  if(not iter_out) return NULL;
  PyObject* py_iterin = iter_in.borrowed();
  PyObject* py_iterout = iter_out.borrowed();

  int const type = PyArray_TYPE((PyArrayObject*)_in);
# ifdef  PYLADA_NPYITER
#   error PYLADA_NPYITER already defined
# endif
# define PYLADA_NPYITER(TYPE, NUM_TYPE)                                       \
    if(type == NUM_TYPE)                                                    \
    {                                                                       \
      while(PyArray_ITER_NOTDONE(py_iterin))                                \
      {                                                                     \
        *((npy_long*)PyArray_ITER_DATA(py_iterout))                         \
            = math::floor_int<TYPE>(*(TYPE*) PyArray_ITER_DATA(py_iterin)); \
        PyArray_ITER_NEXT(py_iterin);                                       \
        PyArray_ITER_NEXT(py_iterout);                                      \
      }                                                                     \
    }
  PYLADA_NPYITER( npy_float,      NPY_FLOAT)      
  else PYLADA_NPYITER( npy_double,     NPY_DOUBLE     )
  else PYLADA_NPYITER( npy_longdouble, NPY_LONGDOUBLE )
  else if(    type == NPY_INT
           or type == NPY_UINT        
           or type == NPY_LONG        
           or type == NPY_LONGLONG    
           or type == NPY_ULONGLONG   
           or type == NPY_BYTE        
           or type == NPY_SHORT       
           or type == NPY_USHORT     ) Py_RETURN_TRUE;
  else
  {
    PYLADA_PYERROR(TypeError, "Unknown numpy array type.");
    return NULL;
  }
# undef PYLADA_NPYITER
  return result.release();
}

PyObject* Rotation1( PyObject *_module, 
                            PyObject *_args, 
                            PyObject *_kwargs )
{
  double angle;
  PyObject *_vector;
  static char *kwlist[] = { const_cast<char*>("angle"),
                            const_cast<char*>("direction"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "dO:Rotation",
                                      kwlist, &angle, &_vector ) )
      return NULL;
  rVector3d vector;
  if(not python::numpy::convert_to_vector(_vector, vector)) return NULL;
  
# ifndef PYLADA_WITH_EIGEN3 
    // \typedef type of the affine transformations.
    typedef Eigen::Transform<types::t_real, 3> Affine;
# else
    // \typedef type of the affine transformations.
    typedef Eigen::Transform<types::t_real, 3, Eigen::Isometry> Affine;
# endif
  // \typedef type of the angle axis object to initialize roations.
  typedef Eigen::AngleAxis<types::t_real> AngleAxis;
  npy_intp dims[2] = {4, 3};
  int const type = python::numpy::type<types::t_real>::value;
  PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(2, dims, type, 1);
  if(not result) return NULL;
  
  Affine a;
  a = AngleAxis(angle, vector);
  for(size_t i(0); i < 3; ++i)
    for(size_t j(0); j < 3; ++j)
      *((types::t_real*)PyArray_GETPTR2(result, i, j)) = a(i, j);
  for(size_t j(0); j < 3; ++j)
    *((types::t_real*)PyArray_GETPTR2(result, 3, j)) = 0;
  return (PyObject*)result;
}
PyObject *translation(PyObject *_module, PyObject *_args)
{
# ifndef PYLADA_WITH_EIGEN3 
    // \typedef type of the affine transformations.
    typedef Eigen::Transform<types::t_real, 3> Affine;
# else
    // \typedef type of the affine transformations.
    typedef Eigen::Transform<types::t_real, 3, Eigen::Isometry> Affine;
# endif
  // \typedef type of the angle axis object to initialize roations.
  npy_intp dims[2] = {4, 3};
  int const type = python::numpy::type<types::t_real>::value;
  PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(2, dims, type, 1);
  if(not result) return NULL;
  
  rVector3d trans;
  if(not python::numpy::convert_to_vector(_args, trans)) return NULL;
  for(size_t i(0); i < 3; ++i)
    for(size_t j(0); j < 3; ++j)
      *((types::t_real*)PyArray_GETPTR2(result, i, j)) = i == j? 1: 0;
  for(size_t j(0); j < 3; ++j)
    *((types::t_real*)PyArray_GETPTR2(result, 3, j)) = trans(j);
  return (PyObject*)result;
}

PyObject* pygruber(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject *_cell;
  int itermax = 0;
  double tolerance = types::tolerance;
  static char *kwlist[] = { const_cast<char*>("cell"),
                            const_cast<char*>("itermax"),
                            const_cast<char*>("tolerance"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "O|id:gruber",
                                      kwlist, &_cell, &itermax, &tolerance ) )
      return NULL;
  rMatrix3d cell;
  if(not python::numpy::convert_to_matrix(_cell, cell)) return NULL;

  try
  {
    rMatrix3d result = gruber(cell, itermax, tolerance);
    return python::numpy::wrap_to_numpy(result);
  }
  catch(error::singular_matrix& _e)
  {
    PYLADA_PYERROR(input, "Singular matrix in gruber.");
    return NULL;
  }
  catch(error::infinite_loop& _e)
  {
    PYLADA_PYERROR(internal, "Maximum number of iterations reached in gruber.");
    return NULL;
  }
  catch(...)
  {
    PYLADA_PYERROR(internal, "Error encoutered in smith normal form.");
    return NULL;
  }
}

PyObject* pysmith(PyObject* _module, PyObject* _matrix)
{
  iMatrix3d matrix;
  if(not python::numpy::convert_to_matrix(_matrix, matrix)) return NULL;

  iMatrix3d S, L, R;
  try { smith_normal_form(S, L, matrix, R); }
  catch(error::singular_matrix& _e)
  {
    PYLADA_PYERROR(input, "Cannot compute smith normal form of singular matrix.");
    return NULL;
  }
  catch(...)
  {
    PYLADA_PYERROR(internal, "Error encoutered in smith normal form.");
    return NULL;
  }
  PyObject *result = PyTuple_New(3);
  if(not result) return NULL;
  PyObject *pyS = python::numpy::wrap_to_numpy(S);
  if(not pyS) { Py_DECREF(result); return NULL; }
  PyObject *pyL = python::numpy::wrap_to_numpy(L);
  if(not pyL) { Py_DECREF(result); Py_DECREF(pyS); return NULL; }
  PyObject *pyR = python::numpy::wrap_to_numpy(R);
  if(not pyR) { Py_DECREF(result); Py_DECREF(pyS); Py_DECREF(pyL); return NULL; }
  PyTuple_SET_ITEM(result, 0, pyS);
  PyTuple_SET_ITEM(result, 1, pyL);
  PyTuple_SET_ITEM(result, 2, pyR);
  return result;
}
   
//! Methods table for math module.
static PyMethodDef methods_table[] = {
    {"is_integer",  pyis_integer, METH_O,
     "True if the input vector or matrix is integer.\n\n"
     "Takes a numpy array as input.\n" }, 
    {"Rotation",  (PyCFunction)Rotation1, METH_KEYWORDS,
     "Rotation of given angle and axis as a 4x3 symmetry operation.\n\n"
     ":param float angle:\n"
     "   Angle of rotation around the input axis.\n"
     ":param axis:\n"
     "   Axis of rotation.\n"
     ":type axis:\n"
     "   sequence of three numbers.\n" }, 
    {"Translation",  (PyCFunction)translation, METH_O,
     "Translation as a 4x3 symmetry operation.\n\n"
     "Takes a sequence of three numbers as input.\n" }, 
    {"floor_int", (PyCFunction)pyfloor_int, METH_O, 
     "Floors an input array.\n\n"
     "Takes a numpy array as input. Returns an integer array.\n" },
    {"gruber", (PyCFunction)pygruber, METH_KEYWORDS,
     "Determines Gruber cell of an input cell.\n\n"
     "The Gruber cell is an optimal parameterization of a lattice, eg shortest\n"
     "cell-vectors and angles closest to 90 degrees.\n\n"
     ":param cell:\n  The input lattice cell-vectors.\n"
     ":type cell:  numpy 3x3 array\n"
     ":param int itermax:\n  Maximum number of iterations. Defaults to 0, ie infinite.\n"
     ":param float tolerance:\n  Tolerance parameter when comparing real numbers. "
        "Defaults to Pylada internals.\n"
     ":returns: An equivalent standardized cell.\n"
     ":raises pylada.error.input: If the input matrix is singular.\n"
     ":raises pylada.error.internal: If the maximum number of iterations is reached.\n"},
     {"smith_normal_form", (PyCFunction)pysmith, METH_O,
      "Computes smith normal form of a matrix.\n\n"
      "If ``M`` is the input matrix, then the smith normal form is\n"
      ":math:`S = L\\cdot M \\cdot R`, with ``S`` diagonal and ``L``\n"
      "and ``R`` are diagonal.\n\nThis function takes an *integer* matrix\n"
      "as its only input.\n\n"
      ":returns: the tuple of matrices (``S``, ``L``, ``R``)." },
    {NULL, NULL, 0, NULL}        /* Sentinel */
}; // end of static method table.


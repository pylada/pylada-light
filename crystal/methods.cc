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

#ifdef PYLADA_MACRO 
#  error PYLADA_MACRO ALREADY defined.
#endif
#define PYLADA_MACRO(NAME)                                                                         \
 /** Wrapper around NAME function */                                                               \
 PyObject* NAME ## _wrapper(PyObject *_module, PyObject *_args)                                    \
 {                                                                                                 \
   Py_ssize_t const N = PyTuple_Size(_args);                                                       \
   if(N != 3 and N != 2)                                                                           \
   {                                                                                               \
     PYLADA_PYERROR(TypeError, #NAME " expects two vectors and a matrix as input.");               \
     return NULL;                                                                                  \
   }                                                                                               \
   math::rMatrix3d cell, invcell;                                                                  \
   if(not python::numpy::convert_to_matrix(PyTuple_GET_ITEM(_args, 1), cell)) return NULL;         \
   if(N == 3 and not python::numpy::convert_to_matrix(PyTuple_GET_ITEM(_args, 2), invcell))        \
     return NULL;                                                                                  \
   if(N == 2) invcell = cell.inverse();                                                            \
   PyObject *positions = PyTuple_GET_ITEM(_args, 0);                                               \
   if(not PyArray_Check(positions))                                                                \
   {                                                                                               \
     math::rVector3d a;                                                                            \
     if(not python::numpy::convert_to_vector(positions, a)) return NULL;                           \
     try                                                                                           \
     {                                                                                             \
       math::rVector3d vector = NAME(a, cell, invcell);                                            \
       npy_intp dim[1] = {3};                                                                      \
       PyObject* result = PyArray_SimpleNew(1, dim,                                                \
           python::numpy::type<math::rVector3d::Scalar>::value );                                  \
       if(result == NULL) return NULL;                                                             \
       *((math::rVector3d::Scalar*)PyArray_GETPTR1((PyArrayObject*)result, 0)) = vector[0];        \
       *((math::rVector3d::Scalar*)PyArray_GETPTR1((PyArrayObject*)result, 1)) = vector[1];        \
       *((math::rVector3d::Scalar*)PyArray_GETPTR1((PyArrayObject*)result, 2)) = vector[2];        \
       return result;                                                                              \
     }                                                                                             \
     /* catch exceptions. */                                                                       \
     catch(std::exception &_e)                                                                     \
     {                                                                                             \
       if(not PyErr_Occurred())                                                                    \
       {                                                                                           \
         PYLADA_PYERROR(internal, ( std::string("Caught c++ exception: ")                          \
                                  + _e.what()).c_str());                                           \
       }                                                                                           \
       return NULL;                                                                                \
     }                                                                                             \
     catch(...)                                                                                    \
     {                                                                                             \
       if(not PyErr_Occurred())                                                                    \
       {                                                                                           \
         PYLADA_PYERROR(internal, "Caught c++ exception: ");                                       \
       }                                                                                           \
       return NULL;                                                                                \
     }                                                                                             \
   }                                                                                               \
   else                                                                                            \
   {                                                                                               \
     int const ndim = PyArray_NDIM((PyArrayObject*)positions);                                     \
     npy_intp *shape = PyArray_DIMS((PyArrayObject*)positions);                                    \
     if(shape[ndim-1] != 3)                                                                        \
     {                                                                                             \
       PYLADA_PYERROR( ValueError,                                                                 \
                     "Last dimension of input numpy array is not of length 3.");                   \
       return NULL;                                                                                \
     }                                                                                             \
     python::Object result(PyArray_SimpleNew(ndim, shape,                                          \
         python::numpy::type<math::rVector3d::Scalar>::value ));                                   \
     if(not result) return NULL;                                                                   \
     python::Object iter_pos( (PyObject*) PyArray_IterNew(positions) );                            \
     python::Object iter_res( (PyObject*) PyArray_IterNew(result.borrowed()) );                    \
     math::rVector3d a;                                                                            \
     int const type = PyArray_TYPE((PyArrayObject*)positions);                                     \
     while(PyArray_ITER_NOTDONE(iter_pos.borrowed()))                                              \
     {                                                                                             \
       /* Shouldn't need to check since last axis is of length == 3 */                             \
       for(size_t i(0); i < 3; ++i)                                                                \
       {                                                                                           \
         void const * const data = PyArray_ITER_DATA(iter_pos.borrowed());                         \
         a(i) = python::numpy::cast_data<types::t_real>(data, type);                               \
         PyArray_ITER_NEXT(iter_pos.borrowed());                                                   \
       }                                                                                           \
                                                                                                   \
       /* Perform action */                                                                        \
       math::rVector3d const vector = NAME(a, cell, invcell);                                      \
                                                                                                   \
       /* copies result */                                                                         \
       for(size_t i(0); i < 3; ++i)                                                                \
       {                                                                                           \
         *( (python::numpy::type<math::rVector3d::Scalar>::np_type*)                               \
             PyArray_ITER_DATA(iter_res.borrowed()) )                                              \
           = (python::numpy::type<math::rVector3d::Scalar>::np_type) vector(i);                    \
         PyArray_ITER_NEXT(iter_res.borrowed());                                                   \
       }                                                                                           \
     }                                                                                             \
     return result.release();                                                                      \
   }                                                                                               \
   return NULL;                                                                                    \
 }
 PYLADA_MACRO(into_cell)
 PYLADA_MACRO(into_voronoi)
 PYLADA_MACRO(zero_centered)
#undef PYLADA_MACRO
#ifdef PYLADA_CATCH
#  error PYLADA_CATCH macro already defined
#endif
#define PYLADA_CATCH                                                              \
 catch(std::exception &_e)                                                      \
 {                                                                              \
   if(not PyErr_Occurred())                                                     \
   {                                                                            \
     PYLADA_PYERROR(internal, ( std::string("Caught c++ exception: ")             \
                              + _e.what()).c_str());                            \
   }                                                                            \
   return NULL;                                                                 \
 }                                                                              \
 catch(...)                                                                     \
 {                                                                              \
   if(not PyErr_Occurred())                                                     \
   {                                                                            \
     PYLADA_PYERROR(internal, "Caught c++ exception: ");                          \
   }                                                                            \
   return NULL;                                                                 \
 }
      
//! Wrapper around periodic image tester.
PyObject* are_periodic_images_wrapper(PyObject *_module, PyObject *_args)
{
  Py_ssize_t const N = PyTuple_Size(_args);
  if(N != 3 and N != 4)
  {
    PYLADA_PYERROR(TypeError, "are_periodic_images expects two vectors and a matrix as input.");
    return NULL;
  }
  math::rMatrix3d invcell;
  math::rVector3d a, b;
  if(not python::numpy::convert_to_vector(PyTuple_GET_ITEM(_args, 0), a)) return NULL;
  if(not python::numpy::convert_to_vector(PyTuple_GET_ITEM(_args, 1), b)) return NULL;
  if(not python::numpy::convert_to_matrix(PyTuple_GET_ITEM(_args, 2), invcell)) return NULL;
  try
  { 
    if(N == 3 and math::are_periodic_images(a, b, invcell) ) Py_RETURN_TRUE;
    else if(N == 4)
    {
      types::t_real tolerance = types::tolerance;
      PyObject* o = PyTuple_GET_ITEM(_args, 3);
      if(PyFloat_Check(o)) tolerance = PyFloat_AS_DOUBLE(o);
      else if(PyInt_Check(o)) tolerance = PyInt_AS_LONG(o);
      else
      {
        PYLADA_PYERROR(TypeError, "Tolerance should be a number.");
        return NULL;
      }
      if(math::are_periodic_images(a, b, invcell, tolerance)) Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  }
  PYLADA_CATCH;
  return NULL;
}
//! Wrapper around the supercell method.
PyObject* supercell_wrapper(PyObject *_module, PyObject *_args, PyObject *_kwargs)
{
  // check/convert input parameters.
  PyObject* lattice;
  PyObject* cellin;
  static char *kwlist[] = { const_cast<char*>("lattice"),
                            const_cast<char*>("supercell"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OO:supercell", kwlist,
                                      &lattice, &cellin) )
    return NULL;
  if(not check_structure(lattice))
  {
    PYLADA_PYERROR(TypeError, "Input is not a crystal.Structure object."); 
    return NULL;
  }
  math::rMatrix3d cell;
  if(not python::numpy::convert_to_matrix(cellin, cell)) return NULL;
  // create supercell.
  try { return supercell(Structure::acquire(lattice), cell).release(); } 
  PYLADA_CATCH;
  return NULL;
}

PyObject* primitive_wrapper(PyObject* _module, PyObject *_args)
{
  Py_ssize_t const N(PyTuple_Size(_args));
  if(N != 1 and N != 2)
  {
    PYLADA_PYERROR(TypeError, "primitive expects a structure as input, and optionally a tolerance.");
    return NULL;
  }
  if(not check_structure(PyTuple_GET_ITEM(_args, 0)))
  {
    PYLADA_PYERROR(TypeError, "First input argument to primitive is not a structure.");
    return NULL;
  }
  Structure const lattice = Structure::acquire(PyTuple_GET_ITEM(_args, 0));
  PyObject *tolobj = N == 2? PyTuple_GET_ITEM(_args, 1): NULL;
  if(N == 2 and (PyInt_Check(tolobj) == false and PyFloat_Check(tolobj) == false))
  {
    PYLADA_PYERROR(TypeError, "First input argument to primitive is not a structure.");
    return NULL;
  }
  types::t_real tolerance = N == 1 ? -1:
          ( PyInt_Check(tolobj) ? PyInt_AS_LONG(tolobj): PyFloat_AS_DOUBLE(tolobj) );
  try { return primitive(lattice, tolerance).release(); } 
  PYLADA_CATCH;
  return NULL;
};

PyObject* is_primitive_wrapper(PyObject* _module, PyObject *_args)
{
  Py_ssize_t const N(PyTuple_Size(_args));
  if(N != 1 and N != 2)
  {
    PYLADA_PYERROR(TypeError, "primitive expects a structure as input, and optionally a tolerance.");
    return NULL;
  }
  if(not check_structure(PyTuple_GET_ITEM(_args, 0)))
  {
    PYLADA_PYERROR(TypeError, "First input argument to primitive is not a structure.");
    return NULL;
  }
  Structure const lattice = Structure::acquire(PyTuple_GET_ITEM(_args, 0));
  PyObject *tolobj = N == 2? PyTuple_GET_ITEM(_args, 1): NULL;
  if(N == 2 and (PyInt_Check(tolobj) == false and PyFloat_Check(tolobj) == false))
  {
    PYLADA_PYERROR(TypeError, "First input argument to primitive is not a structure.");
    return NULL;
  }
  types::t_real tolerance = N == 1 ? -1:
          ( PyInt_Check(tolobj) ? PyInt_AS_LONG(tolobj): PyFloat_AS_DOUBLE(tolobj) );
  try
  {
    if(is_primitive(lattice, tolerance)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
  } 
  PYLADA_CATCH;
  return NULL;
};

PyObject* cell_invariants_wrapper(PyObject *_module, PyObject *_args)
{
  Py_ssize_t const N(PyTuple_Size(_args));
  if(N != 1 and N != 2)
  {
    PYLADA_PYERROR(TypeError, "cell_invariants expects one or two arguments only.");
    return NULL;
  }
  PyObject * const arg0 = PyTuple_GET_ITEM(_args, 0);
  math::rMatrix3d cell;
  if(check_structure(arg0)) cell = ((PyStructureObject*)arg0)->cell;
  else if(not python::numpy::convert_to_matrix(arg0, cell)) return NULL;
  types::t_real tolerance = types::tolerance;
  if(N == 2)
  {
    PyObject * const arg1 = PyTuple_GET_ITEM(_args, 1);
    if(PyInt_Check(arg1)) tolerance = PyInt_AS_LONG(arg1);
    else if(PyFloat_Check(arg1)) tolerance = PyFloat_AS_DOUBLE(arg1);
    else
    {
      PYLADA_PYERROR(TypeError, "Second argument to cell_invariants should a real number.");
      return NULL;
    }
  }
  return cell_invariants(cell, tolerance);
}

PyObject* space_group_wrapper(PyObject *_module, PyObject *_args)
{
  Py_ssize_t const N(PyTuple_Size(_args));
  if(N != 1 and N != 2)
  {
    PYLADA_PYERROR(TypeError, "space_group expects one or two arguments only.");
    return NULL;
  }
  PyObject * const arg0 = PyTuple_GET_ITEM(_args, 0);
  if(not check_structure(arg0)) 
  {
    PYLADA_PYERROR(TypeError, "space_group expects a Structure as first argument.");
    return NULL;
  }
  Structure structure = Structure::acquire(arg0); 
  types::t_real tolerance = types::tolerance;
  if(N == 2)
  {
    PyObject * const arg1 = PyTuple_GET_ITEM(_args, 1);
    if(PyInt_Check(arg1)) tolerance = PyInt_AS_LONG(arg1);
    else if(PyFloat_Check(arg1)) tolerance = PyFloat_AS_DOUBLE(arg1);
    else
    {
      PYLADA_PYERROR(TypeError, "Second argument to cell_invariants should a real number.");
      return NULL;
    }
  }
  return space_group(structure, tolerance);
}

PyObject* equivalent_wrapper(PyObject *_module, PyObject *_args, PyObject *_kwargs)
{
  PyObject *scale = Py_True; 
  PyObject *cartesian = Py_True;
  types::t_real tolerance = types::tolerance;
  PyObject *a = NULL;
  PyObject *b = NULL;
  static char *kwlist[] = { const_cast<char*>("a"),
                            const_cast<char*>("b"), 
                            const_cast<char*>("scale"), 
                            const_cast<char*>("cartesian"), 
                            const_cast<char*>("tolerance"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OO|OOd:equivalent", kwlist,
                                      &a, &b, &scale, &cartesian, &tolerance) )
    return NULL;
  if(not check_structure(a))
  {
    PYLADA_PYERROR(TypeError, "equivalent: First argument should be a structure.");
    return NULL;
  }
  if(not check_structure(b))
  {
    PYLADA_PYERROR(TypeError, "equivalent: second argument should be a structure.");
    return NULL;
  }
  if(scale and not PyBool_Check(scale))
  {
    PYLADA_PYERROR(TypeError, "equivalent: scale should be True or False.");
    return NULL;
  }
  if(cartesian and not PyBool_Check(cartesian))
  {
    PYLADA_PYERROR(TypeError, "equivalent: cartesian should be True or False.");
    return NULL;
  }
  try
  {
    if( equivalent( Structure::acquire(a), Structure::acquire(b), 
                    scale == Py_True ? true: false,
                    cartesian == Py_True ? true: false,
                    tolerance ) ) Py_RETURN_TRUE;
    else if(not PyErr_Occurred()) Py_RETURN_FALSE;
  }
  PYLADA_CATCH;
  return NULL;
}

PyObject* transform_wrapper(PyObject *_module, PyObject *_args, PyObject *_kwargs)
{
  PyObject *transform_ = NULL; 
  PyObject *structure_ = NULL;
  static char *kwlist[] = { const_cast<char*>("structure"),
                            const_cast<char*>("operation"), NULL};
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OO:transform", kwlist,
                                      &structure_, &transform_) )
    return NULL;
  if(not check_structure(structure_))
  {
    PYLADA_PYERROR(TypeError, "structure: First argument should be a structure.");
    return NULL;
  }
  Eigen::Matrix<types::t_real, 4, 3> transform;
  if(not python::numpy::convert_to_matrix(transform_, transform)) return NULL;
  Structure structure = Structure::acquire(structure_).copy();
  structure.transform(transform);
  return structure.release();
 }

//! \brief Wrapper to python for neighbor list creation.
//! \see  Pylada::crystal::neighbors()
PyObject* pyneighbors(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject* structure = NULL; 
  Py_ssize_t nmax = 0;
  PyObject* _center = NULL;
  double tolerance = 1e-8;
  static char *kwlist[] = { const_cast<char*>("structure"),
                            const_cast<char*>("nmax"), 
                            const_cast<char*>("center"), 
                            const_cast<char*>("tolerance"), NULL };
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OIO|d:neighbors", kwlist,
                                      &structure, &nmax, &_center, &tolerance ) )
    return NULL;
  if(not check_structure(structure)) 
  {
    PYLADA_PYERROR(TypeError, "neighbors: First argument should be a structure.");
    return NULL;
  }
  math::rVector3d center(0,0,0);
  if(check_atom(_center)) center = ((PyAtomObject*)_center)->pos;
  else if(not python::numpy::convert_to_vector(_center, center)) return NULL;
  Structure struc = Structure::acquire(structure);
  try { return neighbors(struc, nmax, center, tolerance); }
  catch(...) {}
  return NULL;
}

//! \brief Wrapper to python for coordination_shells list creation.
//! \see  Pylada::crystal::coordination_shells()
PyObject* pycoordination_shells(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject* structure = NULL; 
  Py_ssize_t nmax = 0;
  PyObject* _center = NULL;
  double tolerance = 1e-8;
  Py_ssize_t natoms = 0;
  static char *kwlist[] = { const_cast<char*>("structure"),
                            const_cast<char*>("nshells"), 
                            const_cast<char*>("center"), 
                            const_cast<char*>("tolerance"),
                            const_cast<char*>("natoms"), NULL };
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OIO|dI:coordination_shells", kwlist,
                                      &structure, &nmax, &_center, &tolerance, &natoms ) )
    return NULL;
  if(not check_structure(structure)) 
  {
    PYLADA_PYERROR(TypeError, "coordination_shells: First argument should be a structure.");
    return NULL;
  }
  math::rVector3d center(0,0,0);
  if(check_atom(_center)) center = ((PyAtomObject*)_center)->pos;
  else if(not python::numpy::convert_to_vector(_center, center)) return NULL;
  Structure struc = Structure::acquire(structure);
  try { return coordination_shells(struc, nmax, center, tolerance, natoms); }
  catch(...)
  {
    if(not PyErr_Occurred())
      { PYLADA_PYERROR(InternalError, "Unknown c++ exception occurred.\n"); }
  }
  return NULL;
}
//! \brief Wrapper to python for split configuration creation.
//! \see  Pylada::crystal::splitconfigs()
PyObject* pysplitconfigs(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject* _structure = NULL; 
  PyObject* _atom = NULL; 
  PyObject* _configurations = NULL; 
  Py_ssize_t nmax = 0;
  double tolerance = 1e-8;
  static char *kwlist[] = { const_cast<char*>("structure"),
                            const_cast<char*>("center"), 
                            const_cast<char*>("nmax"), 
                            const_cast<char*>("configurations"), 
                            const_cast<char*>("tolerance"), NULL };
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OOI|Od:splitconfigs", kwlist,
                                      &_structure, &_atom, &nmax, &_configurations, &tolerance) )
    return NULL;
  if(not check_structure(_structure)) 
  {
    PYLADA_PYERROR(TypeError, "splitconfigs: structure argument should be a structure.");
    return NULL;
  }
  if(not check_atom(_atom)) 
  {
    PYLADA_PYERROR(TypeError, "splitconfigs: center argument should be an atom.");
    return NULL;
  }
  if(_configurations and not PyList_Check(_configurations)) 
  {
    PYLADA_PYERROR(TypeError, "splitconfigs: configurations argument should be an list.");
    return NULL;
  }
  Structure structure = Structure::acquire(_structure);
  Atom atom = Atom::acquire(_atom);
  python::Object configs = python::Object::acquire(_configurations);
  try
  { 
    if(not splitconfigs(structure, atom, nmax, configs, tolerance)) return NULL;
    return configs.release();
  }
  catch(...)
  {
    if(PyErr_Occurred() == NULL)
      { PYLADA_PYERROR(InternalError, "splitconfigs: Unknown c++ exception occurred.\n"); }
  }
  return NULL;
}

//! \brief Wrapper to python for mapping sites.
//! \see  Pylada::crystal::map_sites()
PyObject* pymapsites(PyObject* _module, PyObject* _args, PyObject *_kwargs)
{
  PyObject* _mapper = NULL; 
  PyObject* _mappee = NULL; 
  PyObject* _cmp = Py_None; 
  double tolerance = 1e-8;
  static char *kwlist[] = { const_cast<char*>("mapper"),
                            const_cast<char*>("mappee"), 
                            const_cast<char*>("cmp"), 
                            const_cast<char*>("tolerance"), NULL };
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OO|Od:", kwlist,
                                      &_mapper, &_mappee, &_cmp, &tolerance) )
    return NULL;
  if(not check_structure(_mapper)) 
  {
    PYLADA_PYERROR(TypeError, "map_sites: mapper argument should be a structure.");
    return NULL;
  }
  if(not check_structure(_mappee)) 
  {
    PYLADA_PYERROR(TypeError, "map_sites: mappee argument should be a structure.");
    return NULL;
  }
  if(_cmp != Py_None and PyCallable_Check(_cmp) == false)
  {
    PYLADA_PYERROR(TypeError, "map_sites: cmp is expected to be None or a callable");
    return NULL;
  }
  Structure mapper = Structure::acquire(_mapper);
  Structure mappee = Structure::acquire(_mappee);
  python::Object cmp = python::Object::acquire(_cmp);
  try
  { 
    if(map_sites(mapper, mappee, _cmp, tolerance)) Py_RETURN_TRUE;
    else if(not PyErr_Occurred()) Py_RETURN_FALSE;
  }
  catch(...)
  {
    if(PyErr_Occurred() == NULL)
    { PYLADA_PYERROR(InternalError, "map_sites: Unknown c++ exception occurred.\n"); }
  }
  return NULL;
}
  
   
//! Methods table for crystal module.
static PyMethodDef methods_table[] = {
    {"zero_centered",  zero_centered_wrapper, METH_VARARGS,
     "Returns Folds vector back to origin.\n\n"
     "This may not be the vector with the smallest possible norm if the cell is\n"
     "very skewed.\n\n"
     ":param a:\n"
     "    A 3d-vector to fold back into the cell. Can also be any numpy array where\n"
     "    its last dimension if of length 3, ie. ``a.shape[-1] == 3``.\n"
     ":param cell:\n"
     "    The cell matrix defining the periodicity.\n"
     ":param invcell:\n"
     "    Optional. The *inverse* of the cell defining the periodicity. It is\n"
     "    computed if not given on input. \n"}, 
    {"into_voronoi",  into_voronoi_wrapper, METH_VARARGS,
     "Returns Folds vector into first Brillouin zone of the input cell.\n\n"
     "This returns the periodic image with the smallest possible norm.\n\n"
     ":param a:\n"
     "    The 3d-vector to fold back into the cell.\n"
     "    Can also be any numpy array where its last dimension if "
       "of length 3, ie. ``a.shape[-1] == 3``.\n" 
     ":param cell:\n"
     "    The cell matrix defining the periodicity.\n"
     ":param invcell:\n"
     "    Optional. The *inverse* of the cell defining the periodicity.\n"
     "    It is computed if not given on input. \n" },
    {"into_cell",  into_cell_wrapper, METH_VARARGS,
     "Returns Folds vector into periodic the input cell.\n\n"
     ":param a:\n"
     "    The 3d-vector to fold back into the cell.\n"
     "    Can also be any numpy array where its last dimension if "
       "of length 3, ie. ``a.shape[-1] == 3``.\n" 
     ":param cell:\n"
     "    The cell matrix defining the periodicity.\n"
     ":param invcell:\n"
     "    Optional. The *inverse* of the cell defining the periodicity. It\n"
     "    is computed if not given on input. \n" },
    {"are_periodic_images",  are_periodic_images_wrapper, METH_VARARGS,
     "Returns True if first two arguments are periodic images according to third.\n\n"
     ":param a:\n"
     "    First 3d-vector for comparison.\n"
     ":param b: sequence of three numbers\n"
     "    Second 3d-vector for comparison.\n"
     ":param invcell:\n"
     "    The *inverse* of the cell defining the periodicity.\n"
     ":param tolerance:\n"
     "    An *optional* floating point defining the tolerance. Defaults to 1e-8." }, 
    {"supercell",  (PyCFunction)supercell_wrapper, METH_VARARGS | METH_KEYWORDS,
     "Creates a supercell of an input lattice.\n\n"
     ":param lattice:\n"
     "    :py:class:`Structure` from which to create the supercell. Cannot\n"
     "    be empty. Must be deepcopiable. \n" ":param cell: Cell in\n"
     "    cartesian coordinates of the supercell.\n\n"
     ":returns: A :py:class:`Structure` representing the supercell.\n"
     "          If ``lattice`` contains an attribute ``name``, then the\n"
     "          result's is set to \"supercell of ...\".  All other\n"
     "          attributes of the lattice are deep-copied to the supercell.\n"
     "          The atoms within the result contain an attribute ``index``\n"
     "          which is an index to the equivalent site within\n"
     "          ``lattice``. Otherwise, the atoms are deep copies of the\n"
     "          lattice sites. \n"},
    {"primitive",  primitive_wrapper, METH_VARARGS,
     "Returns the primitive unit-cell lattice of an input lattice.\n\n"
     ":param lattice:\n"
     "    :class:`Structure` for which to get primitive unit-cell lattice. "
         "Cannot be empty. Must be deepcopiable. \n"
     ":param tolerance:\n"
     "    Optional. Tolerance when comparing positions. Defaults to 1e-8.\n\n"
     ":returns: A new :class:`Structure` representing the primitive lattice. " },
    {"is_primitive",  is_primitive_wrapper, METH_VARARGS,
     "Returns True if the lattice is primitive.\n\n"
     ":param lattice:\n"
     "    :class:`Structure` for which to get primitive unit-cell lattice. "
        "Cannot be empty. Must be deepcopiable. \n"
     ":param tolerance:\n"
     "    Optional. Tolerance when comparing positions. Defaults to 1e-8.\n\n"},
    {"space_group",  space_group_wrapper, METH_VARARGS,
     "Finds and stores point group operations.\n\n"
     "Implementation taken from ENUM_. \n\n"
     ":param structure:\n"
     "    The :class:`Structure` instance for which to find the point group.\n"
     ":param tolerance:\n"
     "    acceptable tolerance when determining symmetries. Defaults to 1e-8.\n"
     ":returns: python list of affine symmetry operations for the given\n"
     "          structure. Each element is a 4x3 numpy array, with the\n"
     "          first 3 rows forming the rotation, and the last row is the\n"
     "          translation.  The affine transform is applied as rotation *\n"
     "          vector + translation.\n"
     ":raises ValueError: if the input  structure is not primitive.\n\n"
     ".. _ENUM: http://enum.sourceforge.net \n\n"},
    {"cell_invariants",  cell_invariants_wrapper, METH_VARARGS,
     "Finds and stores point group operations.\n\n"
     "Rotations are determined from G-vector triplets with the same norm as\n"
     "the unit-cell vectors. Implementation taken from ENUM_.\n\n"
     ":param structure:\n"
     "    The :py:class:`Structure` instance for which to find the space group.\n"
     "    Can also be a 3x3 matrix.\n"
     ":param tolerance:\n"
     "    acceptable tolerance when determining symmetries. Defaults to 1e-8.\n"
     ":returns: python list of affine symmetry operations for the given\n"
     "          structure. Each element is a 4x3 numpy array, with the\n"
     "          first 3 rows forming the rotation, and the last row is the\n"
     "          translation.  The affine transform is applied as rotation *\n"
     "          vector + translation. `cell_invariants` always returns\n"
     "          rotations (translation is zero).\n\n" 
     ".. _ENUM: http://enum.sourceforge.net/ \n\n"},
    {"equivalent", (PyCFunction)equivalent_wrapper, METH_VARARGS | METH_KEYWORDS, 
      "Returns true if two structures are equivalent. \n\n"
      "Two structures are equivalent in a crystallographic sense, e.g.\n"
      "without reference to cartesian coordinates or possible motif\n"
      "rotations which leave the lattice itself invariant. A supercell is\n"
      "*not* equivalent to its lattice, unless it is a trivial supercell.\n"
      "An option is provided to allow comparison within a single reference\n"
      "frame, i.e. per mathematical definition of lattice.\n\n"
      ":param a:\n  :class:`Structure` to compare to b.\n"
      ":param b:\n  :class:`Structure` to compare to a.\n"
      ":param boolean scale:\n"
      "    whether to take the scale into account. Defaults to true.\n"
      ":param cartesian:\n"
      "    whether to take into account differences in cartesian\n"
      "    coordinates. Defaults to true. If False, then comparison is\n"
      "    according to mathematical definition of a lattice. If True,\n"
      "    comparison is according to crystallographic comparison.\n"
      ":param float tolerance:\n"
      "    Tolerance when comparing distances. Defaults to 1e-8.  It is in\n"
      "    the same units as the structures scales, if that is taken into\n"
      "    account, otherwise, it is in the same units as ``a.scale``." },
    {"transform", (PyCFunction)transform_wrapper, METH_VARARGS | METH_KEYWORDS, 
      "Returns a copy of the structure transformed according to affine operation. \n\n"
      ":param structure:\n"
      "   The :class:`Structure` instance to transform.\n"
      ":param operation:\n"
      "    The rotation (applied first) is contained "
          "in the first three row and the translation (applied second) in the last row." },
    {"periodic_dnc", (PyCFunction)pyperiodic_dnc, METH_VARARGS | METH_KEYWORDS, 
      "Creates periodic divide-and-conquer boxes. \n\n"
      "Creates a list of divide-and-conquer boxes for a periodic system.\n"
      "Takes into account periodic images. In order to make sure that a box\n"
      "contains all atoms relevant for a given calculation, it allows for\n"
      "an overlap, so that atoms which are close (say close enough to form\n"
      "a bond) which are not nominally within the are also referenced. In\n"
      "other words, the boxes have fuzzy limits. The return clearly\n"
      "indicates whether an atom is truly within a box or merely sitting\n"
      "close by.\n\n"
      ":param structure: \n"
      "    The :class:`Structure` instance to transform.\n"
      ":param overlap:\n"
      "    Size in units of the structure's scale attribute which determine\n"
      "    atoms which are close but not quite  within a box.\n"
      ":param mesh:\n"
      "    3d-vector defining the mesh parameters from which to determine\n"
      "    box. Give either this or nperbox, but not both.\n"
      ":param nperbox:\n"
      "    Integer number of atoms per box. Defaults to 20.\n"
      "    Give either this or mesh, but not both.\n"
      ":param return_mesh:\n    If True, returns mesh as first item of a two-tuple.\n"
      ":returns: If ``return_mesh`` is false, returns a list of lists of\n"
      "          3-tuples (atom, translation, insmallbox). Each inner list\n"
      "          represents a single divide-and-conquer box (+overlapping\n"
      "          atoms). Each item within that box contains a reference to\n"
      "          an atom, a possible translation to push the atom from its\n"
      "          current position to its periodic image within the box, and\n"
      "          boolean which is true if the atom truly inside the box as\n"
      "          opposed to sitting just outside within the specified\n"
      "          overlap. If ``return_mesh`` is true, then returns a\n"
      "          two-tuple consisting of the mesh and the list of lists\n"
      "          previously described.\n"},
    {"neighbors", (PyCFunction)pyneighbors, METH_VARARGS | METH_KEYWORDS, 
      "Returns list of neighbors to input position \n\n"
      "Creates a list referencing neighbors of a given position in a structure.\n"
      "In order to make this function well defined, it may return more\n"
      "atoms that actually requested. For instance, in an fcc structure\n"
      "with center at the origin, if asked for the 6 first neighbors,\n"
      "actually the first twelve are returned since they are equidistant.\n"
      "The input tolerance is the judge of equidistance.\n\n"
      ":param structure:\n"
      "    :class:`Structure` from which to determine neighbors.\n"
      ":param nmax:\n"
      "    Integer number of first neighbors to search for.\n"
      ":param center:\n"
      "    Position for which to determine first neighbors.\n"
      ":param tolerance:\n"
      "    Tolerance criteria for judging equidistance.\n\n"
      ":returns: A list of 3-tuples. The first item is a refence to the\n"
      "          neighboring atom, the second is the position of its\n"
      "          relevant periodic image *relative* to the center, the\n"
      "          third is its distance from the center.\n"},
    {"coordination_shells", (PyCFunction)pycoordination_shells, METH_VARARGS | METH_KEYWORDS, 
      "Creates list of coordination shells up to given order.\n\n"
      ":param structure:\n"
      "    :class:`Structure` from which to determine neighbors.\n"
      ":param nshells:\n"
      "    Integer number of shells to compute.\n"
      ":param center:\n"
      "    Position for which to determine first neighbors.\n"
      ":param tolerance:\n"
      "    Tolerance criteria for judging equidistance.\n\n"
      ":param natoms:\n"
      "    Integer number of neighbors to consider. Defaults to fcc + some\n"
      "    security.\n\n"
      ":returns: A list of lists of tuples. The outer list is over\n"
      "          coordination shells.  The inner list references the atoms\n"
      "          in a shell.  Each innermost tuple contains a reference to\n"
      "          the atom in question, a vector from the center to the\n"
      "          relevant periodic image of the atom, and finally, the\n"
      "          associated distance." },
    {"splitconfigs", (PyCFunction)pysplitconfigs, METH_VARARGS | METH_KEYWORDS, 
      "Creates a split-configuration for a given structure and atomic origin.\n\n"
      "Split-configurations are a symmetry-agnostic atom-centered "
      "description of a chemical environment [ABMZ]_.\n\n"
      ":param structure:\n"
      "    :class:`Structure` for which to create split-configurations.\n"
      ":param center:\n"
      "    :class:`Atom` at the origin of the configuration.\n"
      ":param nmax:\n"
      "    Integer number of atoms (cutoff) to consider for inclusion in\n"
      "    the split-configuration.\n"
      ":param configurations:\n"
      "    Defaults to None. Object where configurations should be\n"
      "    stored, eg a list of previously returned configurations. There\n"
      "    is no error checking, so do not mix and match.\n"
      ":param tolerance:\n"
      "    Tolerance criteria when comparing distances. \n\n"
      ":returns: A list of splitted configuration. Each item in this list\n"
      "          is itself a view, e.g. a list with two inner items. The\n"
      "          first inner item is an ordered list of references to\n"
      "          atoms. The second inner item is the weight for that\n"
      "          configuration. The references to the atoms are each a\n"
      "          2-tuple consisting of an actual reference to an atom, as\n"
      "          well as the coordinates of that atom in the current\n"
      "          view.\n\n"
      ".. [ABMZ] `d'Avezac, Botts, Mohlenkamp, Zunger, SIAM J. Comput. 30 (2011)`__\n\n"
      ".. __: http://dx.doi.org/10.1137/100805959\n" },
    {"map_sites", (PyCFunction)pymapsites, METH_VARARGS | METH_KEYWORDS, 
      "Map sites from a lattice onto a structure.\n\n"
      "This function finds out which atomic sites in a supercell refer to\n"
      "the sites in a parent lattice. ``site`` attribute are added to the\n"
      "atoms in the ``mappee`` structure. These attributes hold an index to\n"
      "the relevant sites in the mapper.  If a particular atom could not be\n"
      "mapped, then ``site`` is None.\n\n" 
      ":param mapper:\n"
      "    :class:`Structure` instance acting as the parent lattice.\n"
      ":param mappee:\n"
      "    :class:`Structure` instance acting as the supercell.\n"
      ":param cmp:\n"
      "    Can be set to a callable which shall take two atoms as input and\n"
      "    return True if their occupation (and other attributes) are\n"
      "    equivalent.\n"
      ":param tolerance:\n"
      "  Tolerance criteria when comparing distances.\n\n"
      ":returns: True if all sites in mappee where mapped to mapper.\n" },
    {NULL, NULL, 0, NULL}        /* Sentinel */
}; // end of static method table.
#undef PYLADA_CATCH

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

template<class T>
  bool one_nonzero(Eigen::MatrixBase<T> const &_in)
  {
    for(int i(0), n(0); i < _in.size(); ++i)
      if(_in(i) != 0) 
      {
        if(n == 1) return false;
        n = 1;
      }
    return true;
  };
template<class T>
  bool check_nonzero(Eigen::MatrixBase<T> const &_in, size_t _index)
  {
    int n(0);
    for(int i(0); i < _in.rows(); ++i)
      if(_in(i, _index) != 0) 
      {
        if(n == 1) return false;
        n = 1;
      }
    for(int i(0); i < _in.cols(); ++i)
      if(i != (int)_index and _in(_index, i) != 0) 
      {
        if(n == 1) return false;
        n = 1;
      }
    return true;
  };

template<class T>
  void get_min_max(Eigen::MatrixBase<T> const &_in, types::t_int &_max, types::t_int &_min)
  {
    _max = 0;
    for(int k(1); k < _in.size(); ++k)
      if(std::abs(_in(k)) > std::abs(_in(_max))) _max = k;
    types::t_int k(_in.size()-1);
    for(; k >= 0 and _in(k) == 0; --k);
    _min = k;
    for(--k; k >= 0; --k)
      if(_in(k) != 0 and std::abs(_in(k)) < std::abs(_in(_min))) _min = k;

  }

template<class T>
  void check_same_magnitude( Eigen::MatrixBase<T> const &_in,
                             types::t_int &_max, types::t_int &_min, size_t _index )
  {
    if(std::abs(_in(_min, _index)) != std::abs(_in(_max, _index))) return;
    int n0(0), n1(0);
    for(int i(0); i < _in.rows(); ++i)
    {
      if(_in(_max, i)) ++n0;
      if(_in(_min, i)) ++n1;
    }
    if(n0 < n1 or (n0 == n1 and _in(_max, _index) < _in(_max, _index)) ) std::swap(_min, _max);
  }

//! Column 0 to have only one non-zero positive component placed at origin.
template<class T1, class T2>
  void smith_col_impl( Eigen::MatrixBase<T1> &_left,
                       Eigen::MatrixBase<T2> &_smith, 
                       size_t _index )
  {
    while(not one_nonzero(_smith.col(_index)))
    {
      // find min/max elements.
      types::t_int maxelem, minelem;
      get_min_max(_smith.col(_index), maxelem, minelem); 
      check_same_magnitude(_smith, maxelem, minelem, _index);
      // Remove multiple from column.
      types::t_int const multiple = _smith(maxelem, _index) / _smith(minelem, _index);
      _smith.row(maxelem) -= multiple * _smith.row(minelem);
      _left.row(maxelem) -= multiple * _left.row(minelem);
    }
    if(_smith(_index, _index) == 0) 
    {
      int k(0);
      for(; k < _smith.rows() and _smith(k, _index) == 0; ++k);
      if(k == _smith.rows()) BOOST_THROW_EXCEPTION(error::internal());
      _smith.row(k).swap(_smith.row(_index));
      _left.row(k).swap(_left.row(_index));
    }
    if(_smith(_index, _index) < 0)
    {
      _smith.row(_index) *= -1;
      _left.row(_index) *= -1;
    }
  }

//! Row 0 to have one only non-zero and positive component placed at origin.
template<class T2, class T3>
  void smith_row_impl( Eigen::MatrixBase<T2> &_smith,
                       Eigen::MatrixBase<T3> &_right,
                       size_t _index )
  {
    while(not one_nonzero(_smith.row(_index)))
    {
      // find min/max elements.
      types::t_int maxelem, minelem;
      get_min_max(_smith.row(_index), maxelem, minelem); 
      check_same_magnitude(_smith.transpose(), maxelem, minelem, _index);
      // Remove multiple from column.
      types::t_int const multiple = _smith(_index, maxelem) / _smith(_index, minelem);
      _smith.col(maxelem) -= multiple * _smith.col(minelem);
      _right.col(maxelem) -= multiple * _right.col(minelem);
    }
    if(_smith(_index, _index) == 0) 
    {
      int k(0);
      for(; k < _smith.cols() and _smith(_index, k) == 0; ++k);
      if(k == _smith.cols()) BOOST_THROW_EXCEPTION(error::internal());
      _smith.col(k).swap(_smith.col(_index));
      _right.col(k).swap(_right.col(_index));
    }
    if(_smith(_index, _index) < 0)
    {
      _smith.col(_index) *= -1;
      _right.col(_index) *= -1;
    }
  }


//! Makes matrix diagonal. Does not order diagonal values correctly yet.
template<class T1, class T2, class T3>
  void smith_impl_( Eigen::MatrixBase<T1> &_left,
                    Eigen::MatrixBase<T2> &_right,
                    Eigen::MatrixBase<T3> &_smith )
  {
    for(int index(0); index < _smith.rows()-1; ++index)
      do
      {
        smith_col_impl(_left, _smith, index);
        smith_row_impl(_smith, _right, index);

        int maxrow = _smith.rows();
        types::t_int const diag = _smith(index, index);
        types::t_int maxmod = 0;
        for(int i(index+1); i < _smith.rows(); ++i)
          for(int j(index+1); j < _smith.cols(); ++j)
          {
            if(_smith(i, j) % diag == 0) continue;
            else if(maxmod == 0) { maxrow = i; maxmod = std::abs(_smith(i,j) % diag); }
            else if(std::abs(_smith(i,j) % diag) > maxmod)
              { maxrow = i; maxmod = std::abs(_smith(i,j) % diag); }
          }
        if(maxmod != 0) 
        {
          _smith.row(index) += _smith.row(maxrow);
          _left.row(index)  += _left.row(maxrow);
        }
      } while( not check_nonzero(_smith, index) );
    if(_smith(_smith.rows()-1, _smith.cols()-1) < 0)
    {
      _smith.row(_smith.rows()-1) *= -1;
      _left.row(_smith.rows()-1) *= -1;
    }
  }

void smith_normal_form( iMatrix3d& _S, iMatrix3d & _L,
                        const iMatrix3d& _M, iMatrix3d &_R )
{
  if(_M.determinant() == 0) BOOST_THROW_EXCEPTION(error::singular_matrix());
  // set up the system _out = _left * _smith * _right.
  _L  = iMatrix3d::Identity();
  _R = iMatrix3d::Identity();
  _S = _M;
  smith_impl_(_L, _R, _S);
}

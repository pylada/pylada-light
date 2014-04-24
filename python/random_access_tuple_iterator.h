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
  //! \brief Thin wrapper around a python tuple implementing random access stl iterators.
  //! \details No reference count management. The point is that the tuple
  //!          should be owned and stable throughout the life of the iterator. 
  class RATuple_iterator
  {
    public:
      //! \typedef type representing distance between iterators.
      typedef int difference_type;
      //! \typedef type representing returned by dereferencing.
      typedef PyObject* value_type;
      //! \typedef type representing pointer to referee.
      typedef std::vector<value_type>::iterator::pointer pointer;
      //! \typedef type representing reference to referee.
      typedef std::vector<value_type>::iterator::reference reference;
      //! \typedef type representing the random access iterator category.
      typedef std::vector<value_type>::iterator::iterator_category iterator_category;
      //! Initializes a random access iterator.
      RATuple_iterator(PyObject* _in) : object_(_in), index_(PyTuple_GET_SIZE(_in)) {}
      //! Initializes a random access iterator.
      RATuple_iterator(PyObject* _in, difference_type _index) : object_(_in), index_(_index) {}
      //! Copies a python reference.
      RATuple_iterator(RATuple_iterator const &_in) : object_(_in.object_), index_(_in.index_) {}
      //! Deref operator.
      const reference operator*() const
      {
        if(index_ >= (int)PyTuple_GET_SIZE(object_) or index_ < 0)
        {
          PYLADA_PYERROR(IndexError, "List iterator is out of range.");
          BOOST_THROW_EXCEPTION(error::IndexError());
        }
        return PyTuple_GET_ITEM(object_, index_);
      }
      //! Deref operator.
      reference operator*()
      {
        if(index_ >= (int)PyTuple_GET_SIZE(object_) or index_ < 0)
        {
          PYLADA_PYERROR(IndexError, "List iterator is out of range.");
          BOOST_THROW_EXCEPTION(error::IndexError());
        }
        return PyTuple_GET_ITEM(object_, index_);
      }
      //! Deref operator.
      const pointer operator->() const
      {
        if(index_ >= (int)PyTuple_GET_SIZE(object_) or index_ < 0)
        {
          PYLADA_PYERROR(IndexError, "List iterator is out of range.");
          BOOST_THROW_EXCEPTION(error::IndexError());
        }
        return &PyTuple_GET_ITEM(object_, index_);
      }
      //! Deref operator.
      pointer operator->()
      {
        if(index_ >= (int)PyTuple_GET_SIZE(object_) or index_ < 0)
        {
          PYLADA_PYERROR(IndexError, "List iterator is out of range.");
          BOOST_THROW_EXCEPTION(error::IndexError());
        }
        return &PyTuple_GET_ITEM(object_, index_);
      }
      //! Equality operator.
      bool operator==(RATuple_iterator const &_in) const
        { return object_ == _in.object_ and index_ == _in.index_; }
      //! Inequality operator.
      bool operator!=(RATuple_iterator const &_in) const
        { return object_ != _in.object_ or index_ != _in.index_; }
      //! Less than operator.
      bool operator<(RATuple_iterator const &_in) const
        { return index_ < _in.index_; }
      //! Less than or equal operator.
      bool operator<=(RATuple_iterator const &_in) const
        { return index_ <= _in.index_; }
      //! Greater than operator.
      bool operator>(RATuple_iterator const &_in) const
        { return index_ > _in.index_; }
      //! Greater than or equal operator.
      bool operator>=(RATuple_iterator const &_in) const
        { return index_ >= _in.index_; }
      //! Prefix Increment.
      //! Prefix Increment.
      //! Prefix Increment.
      RATuple_iterator& operator++() { ++index_; return *this;}
      //! Postfix Increment.
      RATuple_iterator operator++(int) { RATuple_iterator tmp(*this); ++index_; return tmp;}
      //! Prefix Decrement.
      RATuple_iterator& operator--() { --index_; return *this;}
      //! Postfix Decrement.
      RATuple_iterator operator--(int) { RATuple_iterator tmp(*this); --index_; return tmp;}
      //! Increments.
      void operator+=(difference_type _a) { index_ += _a; }
      //! Decrements.
      void operator-=(difference_type _a) { index_ += _a; }

      //! Increments.
      RATuple_iterator operator+(difference_type _a) { return RATuple_iterator(object_, index_ + _a); }
      //! Decrements.
      RATuple_iterator operator-(difference_type _a) { return RATuple_iterator(object_, index_ - _a); }
      //! Difference between pointers.
      difference_type operator-(RATuple_iterator const &_a) { return index_ - _a.index_; }

    protected:
      //! Python to tuple.
      PyObject* object_;
      //! Current Item.
      difference_type index_;
  };
#endif

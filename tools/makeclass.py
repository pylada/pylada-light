###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
# 
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
# 
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
# 
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
# 
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

""" Creates functionals (classes) from a method. """
def create_initstring(classname, base, method, excludes):
  """ Creates a string defining the __init__ method. """
  from inspect import getargspec
  from pylada.misc import bugLev

  # creates line:  def __init__(self, ...):
  # keywords are deduced from arguments with defaults.
  # others will not be added.
  args = getargspec(method)
  result = "def __init__(self"
  if args.defaults is not None: 
    nargs = len(args.args) - len(args.defaults)
    for key, value in zip(args.args[nargs:], args.defaults):
      if key in excludes: continue
      result += ", {0}={1!r}".format(key, value)
  result += ", copy=None, **kwargs):\n"

  # adds standard doc string.
  result +=\
   "  \"\"\" Initializes {0} instance.\n\n"                                     \
   "     This function is created automagically from\n"                         \
   "     :py:func:`{1.__module__}.{1.func_name}`. Please see that function\n"   \
   "     for the description of its parameters.\n\n"                            \
   "     :param {2.__name__} copy:\n"                                           \
   "         Deep-copies attributes from this instance to the new (derived)\n"  \
   "         object. This parameter makes easy to create meta-functional from\n"\
   "         the most basic wrappers.\n"                                        \
   "  \"\"\"\n".format(classname, method, base)

  # creates line: from copy import deepcopy
  # used by the copy keyword argument below.
  result += "  from copy import deepcopy\n"
  # creates line: super(BASECLASS, self).__init__(...)
  # arguments are taken from BASECLASS.__init__
  result += "  super(self.__class__, self).__init__("
  initargs = getargspec(base.__init__)
  if initargs.args is not None and len(initargs) > 1:
    # first add args without defaults.
    # fails if not present in method's default arguments.
    ninitargs = len(initargs.args) - len(initargs.defaults)
    for i, key in enumerate(initargs.args[1:ninitargs]):
      if key in excludes: 
        raise Exception('Cannot ignore {1} when synthesizing {0}.'.format(classname, key))
      if key not in args.args[nargs:]:
        raise Exception('Could not synthesize {0}. Missing default argument.'.format(classname))
      result += ", {0}".format(key)
  if initargs.defaults is not None and args.defaults is not None: 
    # then add keyword arguments, ignoring thosse that are not in method
    for i, (key, value) in enumerate(zip(initargs.args[nargs:], initargs.defaults)):
      if key in args.args[ninitargs:]: result += ", {0} = {0}".format(key)
  # add a keyword dict if present in initargs
  if initargs.keywords is not None or initargs.defaults is not None: result += ', **kwargs'
  result += ')\n\n'
  # deals with issues on how to print first argument.
  result = result.replace('(, ', '(')

  # create lines: self.attr = value
  # where attr is something in method which is not in baseclass.__init__
  if args.defaults is not None: 
    for key, value in zip(args.args[nargs:], args.defaults):
      if key in excludes or key in initargs.args: continue
      result += "  self.{0} = {0}\n".format(key)

  # create lines which deep-copies base-class attributes to new derived attributes,
  # eg, using copy. Does not include previously set parameters and anything in
  # excludes.
  avoid = set(initargs.args[:ninitargs]) | set(args.args[nargs:]) | set(excludes)
  result += "  if copy is not None:\n"                                  \
                "    avoid = {0!r}\n"                                   \
                "    for key, value in copy.__dict__.iteritems():\n"    \
                "      if key not in avoid and key not in kwargs:\n"    \
                "         setattr(self, key, deepcopy(value))\n"        \
                .format(avoid)
  if bugLev >= 1:
    print 'tools/makeclass: create_initstring: classname: \"%s\"' \
      % (classname,)
    print 'tools/makeclass: create_initstring: method.__module__: \"%s\"' \
      % (method.__module__,)
    print 'tools/makeclass: create_initstring: method.func_name: \"%s\"' \
      % (method.func_name,)
    print 'tools/makeclass: create_initstring: base.__name__: \"%s\"' \
      % (base.__name__,)
    print 'tools/makeclass: create_initstring: ===== result start ====='
    print result
    print 'tools/makeclass: create_initstring: ===== result end ====='

  return result


def create_iter(iter, excludes):
  """ Creates the iterator method. """
  from inspect import getargspec
  from pylada.misc import bugLev

  # make stateless.
  result = "from pylada.tools import stateless, assign_attributes\n"\
           "@assign_attributes(ignore=['overwrite'])\n@stateless\n"
  # creates line:  def iter(self, ...):
  # keywords are deduced from arguments with defaults.
  # others will not be added.
  args = getargspec(iter)
  result += "def iter(self"
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: result += ", {0}".format(key)
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key, value in zip(args.args[nargs:], args.defaults):
      if key in excludes: result += ", {0}={1!r}".format(key, value)
  # then add kwargs.,
  result += ", **kwargs):\n"

  # adds standard doc string.
  doc = iter.__doc__ 
  if doc is not None and '\n' in doc:
    first_line = doc[:doc.find('\n')].rstrip().lstrip()
    result +=\
        "  \"\"\"{0}\n\n"                                                  \
        "     This function is created automagically from "                \
          ":py:func:`{1.func_name} <{1.__module__}.{1.func_name}>`.\n"     \
        "     Please see that function for the description of its parameters.\n"\
        "  \"\"\"\n"\
        .format(first_line, iter)
  # import iterations method
  result += "  from pylada.tools import SuperCall\n"
  result += "  from {0.__module__} import {0.func_name}\n".format(iter)
  # add iteration line:
  result += "  for o in {0.func_name}(SuperCall(self.__class__, self)"     \
            .format(iter)
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: result += ", {0}".format(key)
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[nargs:]:
      if key in excludes: result += ", {0}={0}".format(key)
      else: result += ", {0}=self.{0}".format(key)
  # adds arguments to overloaded function. 
  if args.keywords is not None: result += ", **kwargs"
  result += "): yield o\n"
 
  if bugLev >= 1:
    print 'tools/makeclass: create_iter: ===== result start ====='
    print result
    print 'tools/makeclass: create_iter: ===== result end ====='
  return result


def create_call_from_iter(iter, excludes):
  """ Creates a call method relying on existence of iter method. """
  from inspect import getargspec
  from pylada.misc import bugLev

  # creates line:  def call(self, ...):
  # keywords are deduced from arguments with defaults.
  # others will not be added.
  args = getargspec(iter)
  callargs = ['self']
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: callargs.append(str(key))
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key, value in zip(args.args[nargs:], args.defaults):
      if key in excludes: callargs.append("{0}={1!r}".format(key, value))
  
  # then add kwargs,
  if args.args is None or 'comm' not in args.args: callargs.append('comm=None')
  if args.keywords is not None: callargs.append('**' + args.keywords)
  result = "def __call__({0}):\n".format(', '.join(callargs))
 
  # adds standard doc string.
  doc = iter.__doc__ 
  if doc is not None and '\n' in doc:
    first_line = doc[:doc.find('\n')].rstrip().lstrip()
    result +=                                                                  \
      "  \"\"\"{0}\n\n"                                                        \
      "     This function is created automagically from\n"                     \
      "     :py:func:`{1.__module__}.{1.func_name}`. Please see that \n"       \
      "     function for the description of its parameters.\n\n"               \
      "     :param comm:\n"                                                    \
      "        Additional keyword argument defining how call external\n"       \
      "        programs.\n"                                                    \
      "     :type comm: :py:class:`~pylada.process.mpi.Communicator`\n\n"        \
      "  \"\"\"\n"                                                             \
      .format(first_line, iter)
  # add iteration line:
  iterargs = []
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: iterargs.append("{0}".format(key))
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[nargs:]:
      if key in excludes: iterargs.append("{0}={0}".format(key))
  # adds arguments to overloaded function. 
  if args.args is None or 'comm' not in args.args:
    iterargs.append('comm=comm')
  if args.keywords is not None: iterargs.append("**" + args.keywords)
  result += "  result  = None\n"                                               \
            "  print 'tools/makeclass: create_call_from_iter: comm: ', comm\n" \
            "  for program in self.iter({0}):\n"                               \
            "    print 'tools/makeclass: create_call_from_iter: program: ', program\n" \
            "    print 'tools/makeclass: create_call_from_iter: type(program): ', type(program)\n" \
            "    if getattr(program, 'success', False):\n"                     \
            "      result = program\n"                                         \
            "      continue\n"                                                 \
            "    if not hasattr(program, 'start'):\n"                          \
            "      return program\n"                                           \
            "    program.start(comm)\n"                                        \
            "    program.wait()\n"                                             \
            "  return result".format(', '.join(iterargs))
  if bugLev >= 1:
    print 'tools/makeclass: create_call_from_iter: ===== result start ====='
    print result
    print 'tools/makeclass: create_call_from_iter: ===== result end ====='
  if bugLev >= 5:
    import os, sys, traceback
    #print 'tools/makeclass: create_call_from_iter: ===== start stack trace'
    #traceback.print_stack( file=sys.stdout)
    #print 'tools/makeclass: create_call_from_iter: ===== end stack trace'

  return result


def create_call(call, excludes):
  """ Creates the call method. """
  from inspect import getargspec
  from pylada.misc import bugLev

  # make stateless.
  result = "from pylada.tools import stateless, assign_attributes\n"\
           "@assign_attributes(ignore=['overwrite'])\n@stateless\n"
  # creates line:  def iter(self, ...):
  # keywords are deduced from arguments with defaults.
  # others will not be added.
  args = getargspec(call)
  result += "def __call__(self"
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: result += ", {0}".format(key)
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key, value in zip(args.args[nargs:], args.defaults):
      if key in excludes: result += ", {0}={1!r}".format(key, value)
  # then add kwargs.,
  result += ", **kwargs):\n"

  # adds standard doc string.
  doc = call.__doc__ 
  if doc is not None and '\n' in doc:
    first_line = doc[:doc.find('\n')].rstrip().lstrip()
    result +=\
        "  \"\"\"{0}\n\n"                                                      \
        "     This function is created automagically from "                    \
        "     {1.__module__}.{1.func_name}. Please see that function for the\n"\
        "     description of its parameters.\n\n"                              \
        "  \"\"\"\n"                                                           \
        .format(first_line, call)
    # import iterations method
  result += "  from pylada.tools import SuperCall\n".format(call)
  result += "  from {0.__module__} import {0.func_name}\n".format(call)
  # add iteration line:
  result += "  return {0.func_name}(SuperCall(self.__class__, self)".format(call)
  if args.args is not None and len(args.args) > 1:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[1:nargs]: result += ", {0}".format(key)
  if args.args is not None and len(args.args) > 1:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[nargs:]:
      if key in excludes: result += ", {0}={0}".format(key)
      else: result += ", {0}=self.{0}".format(key)
  result = result.replace('(, ', '(')
  # adds arguments to overloaded function. 
  if args.keywords is not None: result += ", **kwargs"
  result += ")\n"
 
  if bugLev >= 1:
    print 'tools/makeclass: create_call: ===== result start ====='
    print result
    print 'tools/makeclass: create_call: ===== result end ====='
  return result


def makeclass( classname, base, iter=None, call=None,
               doc=None, excludes=None, module=None ):
  """ Creates a class from a function. 
  
      Makes it easy to create a class which works just like the input method.
      This means we don't have to write the boiler plate methods of a class,
      such as `__init__`. Instead, one can focus on writing a function which
      takes a functional and does something special with it, and then at the
      last minute create an actual derived class from the method and the
      functional. It is used for instance in :py:class:`vasp.Relax
      <pylada.vasp.Relax>`. The parameters from the method which have defaults
      become attributes of instances of this class. Instances can be called as
      one would call the base functional, except of course the job of the
      method is done.
      
      :param str classname:
         Name of the resulting class.
      :param type base:
         Base class, e.g. for a method using VASP, this would be
         :py:class:`Vasp <pylada.vasp.Vasp>`.
      :param function iter:
         The iteration version of the method being wrapped into a class, e.g.
         would override :py:meth:`Vasp.iter <pylada.vasp.Vasp.iter>`.  Ignored if
         None.
      :param function call:
         The __call__ version of the method being wrapped into a class, e.g.
         would override :py:meth:`Vasp.__call__ <pylada.vasp.Vasp.__call__>`.
         Ignored if None.
      :param str doc:
         Docstring of the class. Ignored if None.
      :param list excludes:
         List of strings indicating arguments (with defaults) of the methods
         which should *not* be turned into an attribute. If None, defaults to
         ``['structure', 'outdir', 'comm']``.
      :param bool withkword: 
         Whether to include ``**kwargs`` when calling the __init__ method of
         the *base* class. Only effective if the method accepts variable
         keyword arguments in the first place.
      :param str module:
         Name of the module within which this class will reside.

      :return: A new class derived from ``base`` but implementing the methods
        given on input. Furthermore it contains an `Extract` class-attribute
        coming from either ``iter``, ``call``, ``base``, in that order.
  """
  from pylada.misc import bugLev

  basemethod = iter if iter is not None else call
  if basemethod is None:
    raise ValueError('One of iter or call should not be None.')
  if excludes is None: excludes = ['structure', 'outdir', 'comm']

  # dictionary which will hold all synthesized functions.
  funcs = {}

  # creates __init__
  exec create_initstring(classname, base, basemethod, excludes) in funcs
  if iter is not None:
    exec create_iter(iter, excludes) in funcs
  if call is not None: exec create_call(call, excludes) in funcs
  elif iter is not None:
    exec create_call_from_iter(iter, excludes) in funcs

  d = {'__init__': funcs['__init__']}
  if call is not None or iter is not None: d['__call__'] = funcs['__call__']
  if iter is not None: d['iter'] = funcs['iter']
  if doc is not None and len(doc.rstrip().lstrip()) > 0:
    d['__doc__'] = doc + "\n\nThis class was automagically generated by "\
                         ":py:func:`pylada.tools.makeclass`."
  if hasattr(iter, 'Extract'): d['Extract'] = iter.Extract
  elif hasattr(call, 'Extract'): d['Extract'] = call.Extract
  elif hasattr(base, 'Extract'): d['Extract'] = base.Extract
  if module is not None: d['__module__'] = module
  if bugLev >= 1:
    print 'tools/makeclass: makeclass: classname: \"%s\"' % (classname,)
    print 'tools/makeclass: makeclass: base: \"%s\"' % (base,)
    print 'tools/makeclass: makeclass: d: \"%s\"' % (d,)
  return type(classname, (base,), d)



def makefunc(name, iter, module=None):
  """ Creates function from iterable. """
  from inspect import getargspec
  from pylada.misc import bugLev

  # creates header line of function calls.
  # keywords are deduced from arguments with defaults.
  # others will not be added.
  args = getargspec(iter)
  funcstring = "def {0}(".format(name)
  callargs = []
  if args.args is not None and len(args.args) > 0:
    # first add arguments without default (except for first == self).
    nargs = len(args.args) - len(args.defaults)
    for key in args.args[:nargs]: callargs.append(str(key))
  if args.args is not None and len(args.args) > 0:
    # then add arguments with default
    nargs = len(args.args) - len(args.defaults)
    for key, value in zip(args.args[nargs:], args.defaults):
      callargs.append("{0}={1!r}".format(key, value))
  # adds comm keyword if does not already exist.
  if 'comm' not in args.args: callargs.append('comm=None')
  # adds **kwargs keyword if necessary.
  if args.keywords is not None:
    callargs.append('**{0}'.format(args.keywords))
  funcstring = "def {0}({1}):\n".format(name, ', '.join(callargs))

  # adds standard doc string.
  doc = iter.__doc__ 
  if doc is not None and '\n' in doc:
    first_line = doc[:doc.find('\n')].rstrip().lstrip()
    funcstring +=\
        "  \"\"\"{0}\n\n"                                                      \
        "     This function is created automagically from "                    \
        "     {1.__module__}.{1.func_name}. Please see that function for the\n"\
        "     description of its parameters.\n\n"                              \
        "    :param comm:\n"                                                   \
        "        Additional keyword argument defining how call external\n"     \
        "        programs.\n"                                                  \
        "    :type comm: :py:class:`~pylada.process.mpi.Communicator`\n\n"       \
        "  \"\"\"\n"\
        .format(first_line, iter)
  # create function body...
  funcstring += "  from {0.__module__} import {0.func_name}\n"\
                "  print 'tools/makeclass: makefunc: comm: ', comm\n"\
                "  for program in {0.func_name}(".format(iter)
  # ... including detailed call to iterator function.
  iterargs = []
  if args.args is not None and len(args.args) > 0:
    for key in args.args: iterargs.append("{0}".format(key))
  if args.args is None or 'comm' not in args.args: 
    iterargs.append('comm=comm')
  if args.keywords is not None: iterargs.append('**' + args.keywords)
  funcstring += "{0}):\n"                                                      \
                "    if getattr(program, 'success', False):\n"                 \
                "      result = program\n"                                     \
                "      continue\n"                                             \
                "    if not hasattr(program, 'start'): return program\n"       \
                "    program.start(comm)\n"                                    \
                "    program.wait()\n"                                         \
                "  return result".format(', '.join(iterargs))
  if bugLev >= 1:
    print 'tools/makeclass: makefunc: ===== funcstring start ====='
    print funcstring
    print 'tools/makeclass: makefunc: ===== funcstring end ====='

  funcs = {}
  exec funcstring in funcs
  if bugLev >= 1:
    print 'tools/makeclass: makefunc: after call funcstring'
    print 'tools/makeclass: makefunc: funcs: ', funcs

  if module is not None:  funcs[name].__module__ = module
  if bugLev >= 1:
    print 'tools/makeclass: makefunc: name: \"%s\"' % (name,)
    print 'tools/makeclass: makefunc: funcs[name]: \"%s\"' % (funcs[name],)
  return funcs[name]


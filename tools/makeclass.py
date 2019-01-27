###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
#  submit large numbers of jobs on supercomputers. It provides a python interface to physical input,
#  such as crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
#  programs.  It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
#  General Public License as published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

# pylint: disable=deprecated-method,too-many-branches,too-many-arguments,exec-used

""" Creates functionals (classes) from a method. """
import sys
if sys.version_info.major == 2:
    from inspect import getargspec as func_signature

    def __func_name(func):
        return func.func_name

    def __kwargs(initargs):
        return initargs.keywords
else:
    from inspect import getfullargspec as func_signature

    def __func_name(func):
        return func.__name__

    def __kwargs(initargs):
        return initargs.varkw


def create_initstring(classname, base, method, excludes):
    """ Creates a string defining the __init__ method. """
    # creates line:  def __init__(self, ...):
    # keywords are deduced from arguments with defaults.
    # others will not be added.
    args = func_signature(method)
    result = "def __init__(self"
    if args.defaults is not None:
        nargs = len(args.args) - len(args.defaults)
        for key, value in zip(args.args[nargs:], args.defaults):
            if key in excludes:
                continue
            result += ", {0}={1!r}".format(key, value)
    result += ", copy=None, **kwargs):\n"

    # adds standard doc string.
    result +=\
        "  \"\"\" Initializes {0} instance.\n\n"                                     \
        "     This function is created automagically from\n"                         \
        "     :py:func:`{1.__module__}.{3}`. Please see that function\n"   \
        "     for the description of its parameters.\n\n"                            \
        "     :param {2.__name__} copy:\n"                                           \
        "         Deep-copies attributes from this instance to the new (derived)\n"  \
        "         object. This parameter makes easy to create meta-functional from\n"\
        "         the most basic wrappers.\n"                                        \
        "  \"\"\"\n".format(classname, method, base, __func_name(method))

    # creates line: from copy import deepcopy
    # used by the copy keyword argument below.
    result += "  from copy import deepcopy\n"
    # creates line: super(BASECLASS, self).__init__(...)
    # arguments are taken from BASECLASS.__init__
    result += "  super(self.__class__, self).__init__("
    initargs = func_signature(base.__init__)
    if initargs.args is not None and len(initargs) > 1:
        # first add args without defaults.
        # fails if not present in method's default arguments.
        ninitargs = len(initargs.args) - len(initargs.defaults)
        for key in initargs.args[1:ninitargs]:
            if key in excludes:
                raise Exception('Cannot ignore {1} when synthesizing {0}.'.format(classname, key))
            if key not in args.args[nargs:]:
                raise Exception(
                    'Could not synthesize {0}. Missing default argument.'.format(classname))
            result += ", {0}".format(key)
    if initargs.defaults is not None and args.defaults is not None:
        # then add keyword arguments, ignoring thosse that are not in method
        for key, value in zip(initargs.args[nargs:], initargs.defaults):
            if key in args.args[ninitargs:]:
                result += ", {0} = {0}".format(key)
    # add a keyword dict if present in initargs
    keywords = __kwargs(initargs)
    if keywords is not None or initargs.defaults is not None:
        result += ', **kwargs'
    result += ')\n\n'
    # deals with issues on how to print first argument.
    result = result.replace('(, ', '(')

    # create lines: self.attr = value
    # where attr is something in method which is not in baseclass.__init__
    if args.defaults is not None:
        for key, value in zip(args.args[nargs:], args.defaults):
            if key in excludes or key in initargs.args:
                continue
            result += "  self.{0} = {0}\n".format(key)

    # create lines which deep-copies base-class attributes to new derived attributes,
    # eg, using copy. Does not include previously set parameters and anything in
    # excludes.
    avoid = set(initargs.args[:ninitargs]) | set(args.args[nargs:]) | set(excludes)
    result += "  if copy is not None:\n"                                  \
        "    avoid = {0!r}\n"                                   \
        "    for key, value in copy.__dict__.items():\n"        \
        "      if key not in avoid and key not in kwargs:\n"    \
        "         setattr(self, key, deepcopy(value))\n"        \
        .format(avoid)
    return result


def create_iter(iterator, excludes):
    """ Creates the iterator method. """
    # make stateless.
    result = "from pylada.tools import stateless, assign_attributes\n"\
             "@assign_attributes(ignore=['overwrite'])\n@stateless\n"
    # creates line:  def iterator(self, ...):
    # keywords are deduced from arguments with defaults.
    # others will not be added.
    args = func_signature(iter)
    result += "def iter(self"
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            result += ", {0}".format(key)
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key, value in zip(args.args[nargs:], args.defaults):
            if key in excludes:
                result += ", {0}={1!r}".format(key, value)
    # then add kwargs.,
    result += ", **kwargs):\n"

    # adds standard doc string.
    doc = iterator.__doc__
    if doc is not None and '\n' in doc:
        first_line = doc[:doc.find('\n')].rstrip().lstrip()
        result +=\
            "  \"\"\"{0}\n\n"                                                  \
            "     This function is created automagically from "                \
            ":py:func:`{2} <{1.__module__}.{2}>`.\n"     \
            "     Please see that function for the description of its parameters.\n"\
            "  \"\"\"\n"\
            .format(first_line, iterator, __func_name(iterator))
    # import iterations method
    result += "  from pylada.tools import SuperCall\n"
    result += "  from {0.__module__} import {1}\n".format(iterator, __func_name(iterator))
    # add iteration line:
    result += "  for o in {0}(SuperCall(self.__class__, self)"     \
              .format(__func_name(iterator))
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            result += ", {0}".format(key)
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[nargs:]:
            if key in excludes:
                result += ", {0}={0}".format(key)
            else:
                result += ", {0}=self.{0}".format(key)
    # adds arguments to overloaded function.
    keywords = __kwargs(args)
    if keywords is not None:
        result += ", **kwargs"
    result += "): yield o\n"

    return result


def create_call_from_iter(iter, excludes):
    """ Creates a call method relying on existence of iter method. """
    # creates line:  def call(self, ...):
    # keywords are deduced from arguments with defaults.
    # others will not be added.
    args = func_signature(iter)
    callargs = ['self']
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            callargs.append(str(key))
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key, value in zip(args.args[nargs:], args.defaults):
            if key in excludes:
                callargs.append("{0}={1!r}".format(key, value))

    # then add kwargs,
    if args.args is None or 'comm' not in args.args:
        callargs.append('comm=None')
    keywords = __kwargs(args)
    if keywords is not None:
        callargs.append('**' + keywords)
    result = "def __call__({0}):\n".format(', '.join(callargs))

    # adds standard doc string.
    doc = iterator.__doc__
    if doc is not None and '\n' in doc:
        first_line = doc[:doc.find('\n')].rstrip().lstrip()
        result +=                                                                  \
            "  \"\"\"{0}\n\n"                                                        \
            "     This function is created automagically from\n"                     \
            "     :py:func:`{1.__module__}.{2}`. Please see that \n"       \
            "     function for the description of its parameters.\n\n"               \
            "     :param comm:\n"                                                    \
            "        Additional keyword argument defining how call external\n"       \
            "        programs.\n"                                                    \
            "     :type comm: :py:class:`~pylada.process.mpi.Communicator`\n\n"        \
            "  \"\"\"\n"                                                             \
            .format(first_line, iterator, __func_name(iterator))
    # add iteration line:
    iterargs = []
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            iterargs.append("{0}".format(key))
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[nargs:]:
            if key in excludes:
                iterargs.append("{0}={0}".format(key))
    # adds arguments to overloaded function.
    if args.args is None or 'comm' not in args.args:
        iterargs.append('comm=comm')
    keywords = __kwargs(args)
    if keywords is not None:
        iterargs.append("**" + keywords)
    result += "  result  = None\n"                                               \
              "  for program in self.iter({0}):\n"                               \
              "    if getattr(program, 'success', False):\n"                     \
              "      result = program\n"                                         \
              "      continue\n"                                                 \
              "    if not hasattr(program, 'start'):\n"                          \
              "      return program\n"                                           \
              "    program.start(comm)\n"                                        \
              "    program.wait()\n"                                             \
              "  return result".format(', '.join(iterargs))

    return result


def create_call(call, excludes):
    """ Creates the call method. """
    # make stateless.
    result = "from pylada.tools import stateless, assign_attributes\n"\
             "@assign_attributes(ignore=['overwrite'])\n@stateless\n"
    # creates line:  def iter(self, ...):
    # keywords are deduced from arguments with defaults.
    # others will not be added.
    args = func_signature(call)
    result += "def __call__(self"
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            result += ", {0}".format(key)
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key, value in zip(args.args[nargs:], args.defaults):
            if key in excludes:
                result += ", {0}={1!r}".format(key, value)
    # then add kwargs.,
    result += ", **kwargs):\n"

    # adds standard doc string.
    doc = call.__doc__
    if doc is not None and '\n' in doc:
        first_line = doc[:doc.find('\n')].rstrip().lstrip()
        result +=\
            "  \"\"\"{0}\n\n"                                                      \
            "     This function is created automagically from "                    \
            "     {1.__module__}.{2}. Please see that function for the\n"\
            "     description of its parameters.\n\n"                              \
            "  \"\"\"\n"                                                           \
            .format(first_line, call, __func_name(call))
        # import iterations method
    result += "  from pylada.tools import SuperCall\n"
    result += "  from {0.__module__} import {1}\n".format(call, __func_name(call))
    # add iteration line:
    result += "  return {0}(SuperCall(self.__class__, self)".format(__func_name(call))
    if args.args is not None and len(args.args) > 1:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[1:nargs]:
            result += ", {0}".format(key)
    if args.args is not None and len(args.args) > 1:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[nargs:]:
            if key in excludes:
                result += ", {0}={0}".format(key)
            else:
                result += ", {0}=self.{0}".format(key)
    result = result.replace('(, ', '(')
    # adds arguments to overloaded function.
    keywords = __kwargs(args)
    if keywords is not None:
        result += ", **kwargs"
    result += ")\n"

    return result


def makeclass(classname, base, iterator=None, call=None,
              doc=None, excludes=None, module=None):
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
        :param function iterator:
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
    basemethod = iterator if iterator is not None else call
    if basemethod is None:
        raise ValueError('One of iter or call should not be None.')
    if excludes is None:
        excludes = ['structure', 'outdir', 'comm']

    # dictionary which will hold all synthesized functions.
    funcs = {}

    # creates __init__
    exec(create_initstring(classname, base, basemethod, excludes), funcs)
    if iterator is not None:
        exec(create_iter(iterator, excludes), funcs)
    if call is not None:
        exec(create_call(call, excludes), funcs)
    elif iterator is not None:
        exec(create_call_from_iter(iterator, excludes), funcs)

    methods = {'__init__': funcs['__init__']}
    if call is not None or iterator is not None:
        methods['__call__'] = funcs['__call__']
    if iterator is not None:
        methods['iter'] = funcs['iter']
    if doc is not None and doc.rstrip().lstrip():
        methods['__doc__'] = doc + "\n\nThis class was automagically generated by "\
                             ":py:func:`pylada.tools.makeclass`."
    if hasattr(iterator, 'Extract'):
        methods['Extract'] = iterator.Extract
    elif hasattr(call, 'Extract'):
        methods['Extract'] = call.Extract
    elif hasattr(base, 'Extract'):
        methods['Extract'] = base.Extract
    if module is not None:
        methods['__module__'] = module
    return type(classname, (base,), methods)


def makefunc(name, iterator, module=None):
    """ Creates function from iterable. """
    # creates header line of function calls.
    # keywords are deduced from arguments with defaults.
    # others will not be added.
    args = func_signature(iter)
    funcstring = "def {0}(".format(name)
    callargs = []
    if args.args is not None and args.args:
        # first add arguments without default (except for first == self).
        nargs = len(args.args) - len(args.defaults)
        for key in args.args[:nargs]:
            callargs.append(str(key))
    if args.args is not None and args.args:
        # then add arguments with default
        nargs = len(args.args) - len(args.defaults)
        for key, value in zip(args.args[nargs:], args.defaults):
            callargs.append("{0}={1!r}".format(key, value))
    # adds comm keyword if does not already exist.
    if 'comm' not in args.args:
        callargs.append('comm=None')
    # adds **kwargs keyword if necessary.
    keywords = __kwargs(args)
    if keywords is not None:
        callargs.append('**{0}'.format(keywords))
    funcstring = "def {0}({1}):\n".format(name, ', '.join(callargs))

    # adds standard doc string.
    doc = iterator.__doc__
    if doc is not None and '\n' in doc:
        first_line = doc[:doc.find('\n')].rstrip().lstrip()
        funcstring +=\
            "  \"\"\"{0}\n\n"                                                      \
            "     This function is created automagically from "                    \
            "     {1.__module__}.{2}. Please see that function for the\n"\
            "     description of its parameters.\n\n"                              \
            "    :param comm:\n"                                                   \
            "        Additional keyword argument defining how call external\n"     \
            "        programs.\n"                                                  \
            "    :type comm: :py:class:`~pylada.process.mpi.Communicator`\n\n"       \
            "  \"\"\"\n"\
            .format(first_line, iterator, __func_name(iterator))
    # create function body...
    funcstring += "  from {0.__module__} import {1}\n"\
                  "  for program in {1}(".format(iterator, __func_name(iterator))
    # ... including detailed call to iterator function.
    iterargs = []
    if args.args is not None and args.args:
        for key in args.args:
            iterargs.append("{0}".format(key))
    if args.args is None or 'comm' not in args.args:
        iterargs.append('comm=comm')
    keywords = __kwargs(args)
    if keywords is not None:
        iterargs.append('**' + keywords)
    funcstring += "{0}):\n"                                                      \
                  "    if getattr(program, 'success', False):\n"                 \
                  "      result = program\n"                                     \
                  "      continue\n"                                             \
                  "    if not hasattr(program, 'start'): return program\n"       \
                  "    program.start(comm)\n"                                    \
                  "    program.wait()\n"                                         \
                  "  return result".format(', '.join(iterargs))

    funcs = {}
    exec(funcstring, funcs)

    if module is not None:
        funcs[name].__module__ = module
    return funcs[name]

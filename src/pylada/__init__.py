###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to
#  make it easier to submit large numbers of jobs on supercomputers. It
#  provides a python interface to physical input, such as crystal structures,
#  as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs.
#  It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along with
#  PyLaDa.  If not, see <http://www.gnu.org/licenses/>.
###############################

""" Root of all pylada python packages and modules.

    Configuration variables exist here. However, they are added within separate
    files. Which files will depend upon the user.

       - Files located in the config sub-directory where pylada is installed
       - Files located in one of the directories specified by
         :envvar:`PYLADA_CONFIG_DIR`
       - In the user configuration file ~/.pylada

    The files are read in that order. Within a given directory, files are read
    alphabetically. Later files can override previous files. Finally, all and
    any variable which are declared within these files will end up at the root
    of :py:mod:`pylada`. Be nice, don't pollute yourself.

    .. envvar:: PYLADA_CONFIG_DIR

       Environment variable specifying the path(s) to the configuration
       directories.

    For which variables can be set (and, if relevant, which file) see
    pylada.config.
"""
__docformat__ = "restructuredtext en"
__all__ = [
    "load_ipython_extension", "unload_ipython_extension",
    "error", "crystal", "physics", "misc", "tools", "ewald", "decorations",
    "periodic_table", "vasp", "process", "jobfolder", "logger", "espresso", "physics"]

# noqa: E0611, E114
from ._version import version, version_info

def __find_config_files(pattern="*.py", rcfile=False):
    """ Finds configuration files

        Looks for files with a given pattern in the following directory:

        - config subdirectory of the pylada package
        - directory pointed to by the "PYLADA_CONFIG_DIR" environment variable, if it exists
        - in "~/.pylada" if it exist and is a directory
    """
    from os.path import expandvars, expanduser
    from py.path import local
    from os import environ
    filenames = local(__file__).dirpath("config").listdir(fil=pattern, sort=True)
    for envdir in ['PYLADA_CONFIG_DIR', 'LADA_CONFIG_DIR']:
        if envdir in environ:
            configdir = expandvars(expanduser(environ[envdir]))
            filenames += local(configdir).listdir(fil=pattern, sort=True)
    pylada = local(expanduser("~/.pylada"))
    if pylada.isdir():
        filenames += pylada.listdir(fil=pattern, sort=True)
    elif rcfile and pylada.check(file=True):
        filenames += [pylada]
    return filenames


def __exec_config_files(pattern="*.py", rcfile=False, logger=None):
    """ Executes all config files with given pattern """
    global_dict = {"pyladamodules": __all__}
    local_dict = {}
    for filename in __find_config_files(pattern, rcfile):
        if logger != None:
            logger.debug("Reading configuration file %s" % filename)
        exec(compile(filename.read(), str(filename), 'exec'), global_dict, local_dict)

    return {k: v for k, v in local_dict.items() if k[0] != '_'}


def __setup_logger():
    """ Logger is set up before anything else is done """
    from os import environ
    import logging
    import sys
    local_dict = __exec_config_files("logging.py")
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        stream=sys.stdout)
    logging_level = environ.get('LADA_LOGGING_LEVEL', local_dict['logging_level'])
    try:
        level = int(logging_level)
    except:
        pass
    else:
        logging_level = level
    root_logger = local_dict['root_logger']
    logger = logging.getLogger(root_logger)
    if hasattr(logging, 'upper'):
        logging.setLevel(logging_level.upper())
    else:
        logger.setLevel(logging_level)
    for filename in __find_config_files("logging.py"):
        logger.debug("Read configuration file %s" % filename)
    return logger

# import logger first, so we can print config files
logger = __setup_logger()

# does actual config call.
locals().update(__exec_config_files(rcfile=True, logger=logger))

# import submodules
from . import error, crystal, physics, misc, tools, ewald, decorations, periodic_table, vasp, \
    process, jobfolder, logger, espresso

# Make this an IPython module
from .ipython import load_ipython_extension, unload_ipython_extension

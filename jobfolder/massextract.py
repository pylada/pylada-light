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

""" Classes to manipulate output from job-folder calculations. """
__docformat__ = "restructuredtext en"
__all__ = ['MassExtract']
from .extract import AbstractMassExtract

class MassExtract(AbstractMassExtract): 
  """ Collects extraction properties from different folders. 
  
      Collects extractors across executable folders. The results are presented
      as attributes of an instance of :py:class:`MassExtract`, and arranged as
      directory where the key is the name of the job and the value obtained
      from an instance of that job's Extract. This class is set-up to fail
      silently, and hence is of limited use for diagnosis.

      For properties to be forwarded, the functional should have an ``Extract``
      attribute which takes a directory path as argument.
  """

  def __init__(self, path=None, **kwargs):
    """ Initializes extraction object. 
 
        :param str path:
            Pickled job-folder for which to extract stuff. If None, will
            attempt to use the current job-folder.
        :param kwargs:
            Variable length keyword argument passed on to
            :py:meth:`AbstractMassExtract.__init__`.

        Other arguments are passed on to the base class.
    """
    self.__dict__["_jobfolder"] = None
    super(MassExtract, self).__init__(path=path, **kwargs)

  @property
  def jobfolder(self):
    """ Root of the job-folder wrapped by this instance. """
    from . import load
    from .. import is_interactive
    if self._jobfolder is None:
      if self._rootpath is None: 
        if is_interactive:
          from .. import interactive
          if interactive.jobfolder is None:
            print "No current job-dictionary."
            return
          return interactive.jobfolder.root
        else: raise RuntimeError('No job-folder.')
      else: self._jobfolder = load(self.rootpath, timeout=30)
    return self._jobfolder.root

  @property
  def rootpath(self):
    """ Root of the directory tree where computational results can be found. """
    from .. import is_interactive
    if self._jobfolder is None and self._rootpath is None and is_interactive:
      from .. import interactive
      if interactive.jobfolder_path is None:
        print "No current path to job-dictionary."
        return
      return interactive.jobfolder_path
    return super(MassExtract, self).rootpath
    

  def __iter_alljobs__(self):
    """ Generator to go through all relevant jobs.  
    
        :return: (name, extractor), where name is the name of the job, and
          extractor an extraction object.
    """
    from os.path import join, dirname
    
    for name, job in self.jobfolder.iteritems():
      if job.is_tagged: continue
      try: extract = job.functional.Extract(join(dirname(self.rootpath), name))
      except: pass 
      else: yield job.name, extract

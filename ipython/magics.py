""" IPython extension module for Pylada. """
from IPython.core.magic import magics_class, line_magic, Magics

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

__docformat__ = "restructuredtext en"


@magics_class
class Pylada(Magics):
    @line_magic
    def savefolders(self, line):
        from .savefolders import savefolders
        return savefolders(self, line)

    @line_magic
    def explore(self, line):
        from .explore import explore
        return explore(self, line)

    @line_magic
    def goto(self, line):
        from .goto import goto
        return goto(self, line)

    @line_magic
    def listfolders(self, line):
        from .listfolders import listfolders
        return listfolders(self, line)

    @line_magic
    def fl(self, line):
        """ Alias for %listfolders """
        from .listfolders import listfolders
        return listfolders(self, line)

    @line_magic
    def showme(self, line):
        from .showme import showme
        return showme(self, line)

    @line_magic
    def launch(self, line):
        from .launch import launch
        return launch(self, line)

    @line_magic
    def export(self, line):
        from .export import export
        return export(self, line)

    @line_magic
    def copyfolder(self, line):
        from .manipfolders import copy_folder
        return copy_folder(self, line)

    @line_magic
    def deletefolder(self, line):
        from .manipfolders import delete_folder
        return delete_folder(self, line)

    @line_magic
    def qstat(self, arg):
        """ SList of user's jobs.

            The actual print-out and functionality will depend on the
            user-specified function :py:func:`pylada.ipython_qstat`. However,
            in general %qstat should work as follows:

            >>> %qstat
            [ 'id something something jobname' ]

            It returns an SList_ of all the users jobs, with the job-id as the
            first column and the job-name ass the last. The results can be
            filtered using SList_'s grep, or directly as in:

            >>> %qstat hello
            [ 'id something something hellorestofname' ]

            .. _SList: http://ipython.org/ipython-doc/stable/api/generated/IPython.utils.text.html#slist

        """
        import pylada
        if not hasattr(pylada, 'ipython_qstat'):
            pylada.logger.getChild("ipython").warning(
                "Missing ipython_qstat function: cannot use %qstat")
            return []

        ipython_qstat = pylada.ipython_qstat
        arg = arg.rstrip().lstrip()
        if len(arg) != 0 and '--help' in arg.split() or '-h' in arg.split():
            print(self.qstat.__doc__ + '\n' + ipython_qstat.__doc__)
            return None

        result = ipython_qstat(self, arg)
        if len(arg) == 0:
            return result

        return result.grep(arg, field=-1)

    @line_magic
    def qdel(self, arg):
        """ Cancel jobs which grep for whatever is in arg.

            For instance, the following cancels all jobs with "anti-ferro" in
            their name. The name is the last column in qstat.

            >>> %qdel "anti-ferro"
        """
        import six
        import pylada
        from pylada import qdel_exe

        if not hasattr(pylada, 'ipython_qstat'):
            raise RuntimeError(
                "Missing ipython_qstat function: cannot use %qdel")

        arg = arg.lstrip().rstrip()
        if '--help' in arg.split() or '-h' in arg.split():
            print(self.qdel.__doc__)
            return

        if not arg:
            result = self.qstat(arg)
            if not result:
                print('No jobs in queue')
                return
            for name in result.fields(-1):
                print("cancelling %s." % (name))
                message = "Are you sure you want to cancel"\
                    "the jobs listed above? [y/n] "
        else:
            message = "Cancel all jobs? [y/n] "
            key = ''
            while key not in ['n', 'y']:
                key = six.raw_input(message)
                if key == 'n':
                    return

        result = self.qstat(arg)
        for i, name in zip(result.fields(0), result.fields(-1)):
            # xxx use subprocess
            self.shell.system('{0} {1}'.format(qdel_exe, i))

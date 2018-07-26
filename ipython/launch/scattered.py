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

""" Launches scattered calculations.

 
    This launch strategy will send one pbs/slurm job per pylada job.

    >>> %launch scattered --walltime 24:00:00 
"""
__docformat__ = "restructuredtext en"


def launch(self, event, jobfolders):
    """ Launch scattered jobs: one job = one pbs script. """
    from copy import deepcopy
    import os
    import re
    import subprocess
    from os.path import exists, basename
    from os import remove
    import pylada
    from .. import get_shell
    from ...misc import local_path, testValidProgram
    from ... import pbs_string, default_pbs, qsub_exe, default_comm
    from . import get_walltime, get_mppalloc, get_queues, scattered_script
    from .. import logger
    from ..magics import Pylada

    if not hasattr(pylada, 'ipython_qstat'):
        logger.warning("Missing ipython_qstat function: cannot check for jobs already in queue")
        qstat = lambda x: []
    else:
        qstat = lambda x: self.qstat(x)

    logger.info("launch/scattered: event: %s" % event)
    shell = get_shell(self)

    pbsargs = deepcopy(dict(default_comm))
    pbsargs.update(default_pbs)
    pbsargs['ppn'] = event.ppn

    mppalloc = get_mppalloc(shell, event)
    if mppalloc is None:
        return

    # Set pbsargs['walltime'] to a string like '03:59:59'
    if not get_walltime(shell, event, pbsargs):
        return

    # Set pbsargs['queue'], pbsargs['account']
    if not get_queues(shell, event, pbsargs):
        return
    logger.info("launch/scattered: pbsargs: %s" % pbsargs)

    # gets python script to launch in pbs.
    pyscript = scattered_script.__file__
    logger.info("launch/scattered: pyscript: %s" % pyscript)
    if pyscript[-1] == 'c':
        pyscript = pyscript[:-1]   # change .pyc to .py

    # creates file names.
    hasprefix = getattr(event, "prefix", None)

    def pbspaths(directory, jobname, suffix):
        """ creates filename paths. """
        suffix = '{0}-pbs{1}'.format(event.prefix, suffix) if hasprefix \
            else 'pbs{0}'.format(suffix)
        return str(directory.join(jobname, suffix))
    # now  loop over jobfolders
    pbsscripts = []
    for current, path in jobfolders:
        logger.info("launch/scattered: current: %s  path: %s" % (current, path))
        # creates directory.
        directory = local_path(path).dirpath()
        directory.ensure(dir=True)
        # loop over executable folders in current jobfolder
        for name, job in current.root.items():
            logger.info('launch/scattered: current: %s' % current)
            logger.info('launch/scattered: current.root: %s' % current.root)
            logger.info('launch/scattered: name: %s' % name)
            logger.info('launch/scattered: job: %s' % job)
            logger.info('launch/scattered: job.is_tagged: %s' % job.is_tagged)

            # avoid jobfolder which are off
            if job.is_tagged:
                continue

            # added by Peter Graf
            # avoid jobfolder which is already in the queue:
            qstuff = qstat(name)
            if (len(qstuff) > 0 and not event.force):
                status = [x.split()[2] for x in qstuff]
                # status is a list like ['Q'], ['R'], ['H'], ['C'], ['R', 'C'], etc
                # 'RHQ' is the status that the job is indeed in the queue, 'C' job completed and
                # being removed from the queue if needed, a prefix can be used to distinguish two
                # jobs with the same name
                if len(set(status) & set('RHQ')) > 0:
                    print(("Job %s is in the queue, will not be re-queued" % name))
                    continue

            # avoid successful jobs.unless specifically requested
            if hasattr(job.functional, 'Extract') and not event.force:
                p = directory.join(name)
                extract = job.functional.Extract(str(p))
                if extract.success:
                    print(("Job {0} completed successfully. "
                           "It will not be relaunched.".format(name)))
                    continue

            # setup parameters for launching/running jobs
            pbsargs['n'] = mppalloc(job) if hasattr(mppalloc, "__call__")            \
                else mppalloc
            pbsargs['nnodes'] = (pbsargs['n'] + pbsargs['ppn'] - 1)                  \
                // pbsargs['ppn']
            pbsargs['err'] = pbspaths(directory, name, 'err')
            pbsargs['out'] = pbspaths(directory, name, 'out')
            pbsargs['name'] = name if len(name)                                      \
                else "{0}-root".format(basename(path))
            pbsargs['directory'] = str(directory)
            pbsargs['logging'] = 'critical'
            pbsargs['testValidProgram'] = testValidProgram

            pbsargs['scriptcommand']                                                 \
                = "{0} --logging {logging} --testValidProgram {testValidProgram} --nbprocs {n} --ppn {ppn} --jobid={1} {2}"                   \
                .format(pyscript, name, path, **pbsargs)
            ppath = pbspaths(directory, name, 'script')
            logger.info("launch/scattered: ppath: \"%s\"" % ppath)
            logger.info("launch/scattered: pbsargs: \"%s\"" % pbsargs)
            pbsscripts.append(ppath)

            # write pbs scripts
            directory.join(name).ensure(dir=True)
            if exists(pbsscripts[-1]):
                remove(pbsscripts[-1])
            with open(pbsscripts[-1], "w") as file:
                string = pbs_string(**pbsargs) if hasattr(pbs_string, '__call__')      \
                    else pbs_string.format(**pbsargs)
                # peregrine takes back the option of "anynode"
                string = string.replace("#PBS -l feature=anynode", "##PBS -l feature=anynode")
                logger.info(
                    "launch/scattered: ===== start pbsscripts[-1]: %s =====" % pbsscripts[-1])
                logger.info('%s' % string)
                logger.info(
                    "launch/scattered: ===== end pbsscripts[-1]: %s =====" % pbsscripts[-1])
                lines = string.split('\n')
                omitTag = '# omitted for testValidProgram: '
                for line in lines:
                    if testValidProgram != None \
                        and (re.match('^ *module ', line)
                             or re.match('^\. .*/bin/activate$', line)):
                        line = omitTag + line
                    file.write(line + '\n')
            assert exists(pbsscripts[-1])

        print(("Created {0} scattered jobs from {1}.".format(len(pbsscripts), path)))

    if event.nolaunch:
        return
    # otherwise, launch.
    for script in pbsscripts:
        logger.info("launch/scattered: launch: shell: %s" % shell)
        logger.info("launch/scattered: launch: qsub_exe: %s" % qsub_exe)
        logger.info("launch/scattered: launch: script: \"%s\"" % script)

        if testValidProgram != None:
            cmdLine = '/bin/bash ' + script
        else:
            # qsub pbsscript (template is in config/mpi.py: pbs_string),
            # which sets up modules and invokes: python {scriptcommand}
            cmdLine = "{0} {1}".format(qsub_exe, script)

        nmerr = script + '.stderr'
        nmout = script + '.stdout'
        with open(nmerr, 'w') as ferr:
            with open(nmout, 'w') as fout:
                subprocess.call(cmdLine, shell=True, stderr=ferr, stdout=fout)
                # xxx: all subprocess: set stderr, stdout
        if os.path.getsize(nmerr) != 0:
            with open(nmerr) as fin:
                print('launch/scattered: stderr: %s' % (fin.read(),))
        with open(nmout) as fin:
            print('launch/scattered: stdout: %s' % (fin.read(),))


def completer(self, info, data):
    """ Completer for scattered launcher. """
    from .. import jobfolder_file_completer
    from ... import queues, accounts, debug_queue, features
    if len(data) > 0:
        if data[-1] == "--walltime":
            return [u for u in self.user_ns
                    if u[0] != '_' and isinstance(self.user_ns[u], str)]
        elif data[-1] == "--nbprocs":
            result = [u for u in self.user_ns
                      if u[0] != '_' and isinstance(self.user_ns[u], int)]
            result.extend([u for u in self.user_ns
                           if u[0] != '_' and hasattr(u, "__call__")])
            return result
        elif data[-1] == '--ppn':
            return ['']
        elif data[-1] == "--prefix":
            return ['']
        elif data[-1] == "--queue":
            return queues
        elif data[-1] == "--account":
            return accounts
        elif data[-1] == "--feature":
            return features
    result = ['--force', '--walltime', '--nbprocs', '--help']
    if len(queues) > 0:
        result.append("--queue")
    if len(accounts) > 0:
        result.append("--account")
    if len(features) > 0:
        result.append("--feature")
    if debug_queue is not None:
        result.append("--debug")
    result.extend(jobfolder_file_completer([info.symbol]))
    result = list(set(result) - set(data))
    return result


def parser(self, subparsers, opalls):
    """ Adds subparser for scattered. """
    from ... import default_comm
    from . import set_queue_parser, set_default_parser_options
    result = subparsers.add_parser('scattered',
                                   description="A separate PBS/slurm script is created for each "
                                   "and every calculation in the job-folder "
                                   "(or dictionaries).",
                                   parents=[opalls])
    set_default_parser_options(result)
    result.add_argument('--nbprocs', type=str, default="None", dest="nbprocs",
                        help="Can be an integer, in which case it specifies "
                        "the number of processes to exectute jobs with. "
                        "Can also be a callable taking a JobFolder as "
                        "argument and returning a integer. Will default "
                        "to as many procs as there are atoms in that "
                        "particular structure. Defaults to something "
                        "close to the number of atoms in the structure "
                        "(eg good for VASP). ")
    result.add_argument('--ppn', dest="ppn",
                        default=default_comm.get('ppn', 1), type=int,
                        help="Number of processes per node. Defaults to {0}."
                        .format(default_comm.get('ppn', 1)))
    set_queue_parser(result)
    result.set_defaults(func=launch)
    return result

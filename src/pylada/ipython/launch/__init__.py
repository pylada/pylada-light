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

""" IPython launch magic function. """
__docformat__ = "restructuredtext en"


def launch(self, event):
    """ Launches PBS/slurm jobs.


        The usual setup is to first explore a dictionary of some sort, 
        then modify it and save it, and finally launch it.

        >>> # to recompute errors.
        >>> %explore errors path/to/original/pickle
        >>> %goto next
        >>> # modify dictionary here.
        >>> %showme functional
        >>> ...
        >>> %goto next
        >>> %showme functional
        >>> ...
        >>> # Saves the modified job.
        >>> # the new path could a different filename in the same directory.
        >>> # This way, the unsuccessful output from the first run will be
        >>> # overwritten.
        >>> %savejobs path/to/original/pickle.errors
        >>> # then  launch.
        >>> %launch scattered --walltime "24:00:00"
    """
    import argparse
    from ...jobfolder import load as load_jobs
    from ... import interactive, qsub_array_exe
    from .scattered import parser as scattered_parser
    from .interactive import parser as interactive_parser
    from .asone import parser as asone_parser
    from .array import parser as array_parser
    from .single import parser as single_parser
    from ...misc import RelativePath, LockFile

    # main parser
    parser = argparse.ArgumentParser(prog='%launch')
    # options supported by all.
    opalls = argparse.ArgumentParser(add_help=False)
    opalls.add_argument('pickle', metavar='FILE', type=str, nargs='*',
                        default="",
                        help='Optional path to a job-folder. If not present, the '
                        'currently loaded job-dictionary will be launched.')
    opalls.add_argument('--force', action="store_true", dest="force",
                        help="If present, launches all untagged jobs, even those "
                        "which completed successfully.")
    # subparsers
    subparsers                                                                   \
        = parser.add_subparsers(help='Launches one job per untagged calculations')

    # launch scattered.
    scattered_parser(self, subparsers, opalls)
    interactive_parser(self, subparsers, opalls)
    asone_parser(self, subparsers, opalls)
    single_parser(self, subparsers, opalls)
    if qsub_array_exe is not None:
        array_parser(self, subparsers, opalls)

    # parse arguments
    try:
        args = parser.parse_args(event.split())
    except SystemExit as e:
        return None

    # creates list of dictionaries.
    jobfolders = []
    if args.pickle != '':
        for pickle in args.pickle:
            pickle = RelativePath(pickle).path
            try:
                d = load_jobs(path=pickle, timeout=20)
            except ImportError as e:
                print("ImportError: ", e)
                return
            except Exception as e:
                print(e)
                if LockFile(pickle).is_locked:
                    print("You may want to check for the existence of {0}."\
                          .format(LockFile(pickle).lock_directory))
                    print("If you are sure there are no jobs out there accessing {0},\n"\
                          "you may want to delete that directory.".format(args.pickle))
                    return
            else:
                jobfolders.append((d, pickle))
    else:  # current job folder.
        if interactive.jobfolder is None:
            print("No current job-dictionary.")
            return
        if interactive.jobfolder_path is None:
            print("No path for currrent job-dictionary.")
            return
        jobfolders = [(interactive.jobfolder, interactive.jobfolder_path)]

    # calls specialized function.
    args.func(self, args, jobfolders)


def completer(self, info):
    """ Completion for launchers. """
    from .scattered import completer as scattered_completer
    from .interactive import completer as interactive_completer
    from .asone import completer as asone_completer
    from .array import completer as array_completer
    from .single import completer as single_completer
    from ... import qsub_array_exe

    data = info.line.split()[1:]
    if "scattered" in data:
        return scattered_completer(self, info, data)
    elif "interactive" in data:
        return interactive_completer(self, info, data)
    elif "asone" in data:
        return asone_completer(self, info, data)
    elif "single" in data:
        return single_completer(self, info, data)
    elif qsub_array_exe is not None and "array" in data:
        return array_completer(self, info, data)
    result = ["scattered", "interactive", 'asone', 'single', '--help']
    return result + ['array'] if qsub_array_exe is not None else result


def get_mppalloc(shell, event, withdefault=True):
    """ Gets mpp allocation. """
    from .. import logger

    logger.info("launch/init: shell: %s" % shell)
    logger.info("launch/init: event: %s" % event)
    logger.info("launch/init: event.ppn: %s" % event.ppn)
    logger.info("launch/init: withdefault: %s" % withdefault)
    try:
        mppalloc = shell.ev(event.nbprocs)
    except Exception as e:
        print(("Could not make sense of --nbprocs argument {0}.\n{1}"               \
              .format(event.nbprocs, e)))
        return
    logger.info("launch/init: mppalloc a: %s" % mppalloc)
    if mppalloc is None and withdefault:
        def mppalloc(job):
            """ Returns number of processes for this job. """
            natom = len(job.structure)  # number of atoms.
            # Round down to a multiple of ppn
#vladan            nnode = max(1, natom / event.ppn)
            nnode = max(1, natom // event.ppn)
            nproc = nnode * event.ppn
            return nproc
    logger.info("launch/init: mppalloc b: %s" % mppalloc)
    return mppalloc


def get_walltime(shell, event, pbsargs):
    """ Returns walltime. """
    from re import match
    if match("\s*(\d{1,3}):(\d{1,2}):(\d{1,2})\s*", event.walltime) is None:
        try:
            walltime = shell.ev(event.walltime)
        except Exception as e:
            print("Could not make sense of --walltime argument {0}.\n{1}"            \
                  .format(event.walltime, e))
            return False
    else:
        walltime = event.walltime
    walltime = match("\s*(\d{1,3}):(\d{1,2}):(\d{1,2})\s*", walltime)
    if walltime is not None:
        a, b, c = walltime.group(1), walltime.group(2), walltime.group(3)
        walltime = "{0:0>2}:{1:0>2}:{2:0>2}".format(a, b, c)
    else:
        print("Could not make sense of --walltime argument {0}."                   \
              .format(event.walltime))
        return False
    pbsargs['walltime'] = walltime
    return True


def get_queues(shell, event, pbsargs):
    """ Decodes queue/account/feature options. """
    from ... import debug_queue
    if event.__dict__.get("queue", None) is not None:
        pbsargs["queue"] = event.queue
    else:
        pbsargs["queue"] = 'batch'   # default queue
    if event.__dict__.get("account", None) is not None:
        pbsargs["account"] = event.account
    if event.__dict__.get("feature", None) is not None:
        pbsargs["feature"] = event.feature
    if getattr(event, 'debug', False):
        if debug_queue is None:
            print("No known debug queue for this machine")
            return False
        pbsargs[debug_queue[0]] = debug_queue[1]
    return True


def set_default_parser_options(parser):
    """ Adds some default options to parser """
    from ... import default_pbs, qsub_exe
    parser.add_argument('--walltime', type=str,
                        default=default_pbs['walltime'],
                        help='walltime for jobs. Should be in hh:mm:ss format. '
                        'Defaults to ' + default_pbs['walltime'] + '.')
    parser.add_argument('--prefix', action="store",
                        type=str, help="Adds prefix to job name.")
    parser.add_argument('--nolaunch', action="store_true",
                        dest="nolaunch",
                        help='Does everything except calling {0}.'.format(qsub_exe))
    return


def set_queue_parser(parser):
    """ Adds default queue/account/feature options. """
    from ... import queues, accounts, debug_queue, features

    parser.add_argument(
        '--account', dest='account', type=str,
        help='Launches jobs on specific account if present.')

    parser.add_argument(
        '--feature', dest='feature', type=str,
        help='Launches jobs on specific feature if present.')

    if len(queues) != 0:
        parser.add_argument('--queue', dest="queue", choices=queues,
                            default=queues[0],
                            help="Queue on which to launch job. Defaults to {0}."
                            .format(queues[0]))
    else:
        parser.add_argument('--queue', dest="queue", type=str,
                            help="Launches jobs on specific queue if present.")
    if debug_queue is not None:
        parser.add_argument('--debug', dest="debug", action="store_true",
                            help="launches in interactive queue if present.")

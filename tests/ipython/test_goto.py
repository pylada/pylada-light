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


def test(shell, tmpdir, functional):
    from os.path import join
    from os import getcwd
    from pylada.jobfolder import JobFolder
    from pylada import interactive

    root = JobFolder()
    for type, trial, size in [
        ("this", 0, 10),
        ("this", 1, 15),
        ("that", 2, 20),
        ("that", 1, 20),
    ]:
        job = root / type / str(trial)
        job.functional = functional
        job.params["indiv"] = size
        if type == "that":
            job.params["value"] = True

    shell.user_ns["jobfolder"] = root
    shell.magic("explore jobfolder")
    shell.magic("savefolders {0}/dict".format(tmpdir))
    for name, job in root.items():
        result = job.compute(outdir=join(tmpdir, name))
        assert result.success
        assert {
            "this/0": 10,
            "this/1": 15,
            "that/1": 20,
            "that/2": 20,
            "this/0/another": 25,
        }[name] == result.indiv

    shell.magic("explore {0}/dict".format(tmpdir))
    shell.magic("goto this/0")
    assert getcwd() == "{0}/this/0".format(tmpdir)
    assert interactive.jobfolder.name == "/this/0/"
    shell.magic("goto ../1")
    assert getcwd() == "{0}/this/1".format(tmpdir)
    assert interactive.jobfolder.name == "/this/1/"
    shell.magic("goto /that")
    assert getcwd() == "{0}/that".format(tmpdir)
    assert interactive.jobfolder.name == "/that/"
    shell.magic("goto 2")
    assert getcwd() == "{0}/that/2".format(tmpdir)
    assert interactive.jobfolder.name == "/that/2/"
    shell.magic("goto /")
    shell.magic("goto next")
    assert getcwd() == "{0}/that/1".format(tmpdir)
    assert interactive.jobfolder.name == "/that/1/"
    shell.magic("goto next")
    assert getcwd() == "{0}/that/2".format(tmpdir)
    assert interactive.jobfolder.name == "/that/2/"
    shell.magic("goto previous")
    assert getcwd() == "{0}/that/1".format(tmpdir)
    assert interactive.jobfolder.name == "/that/1/"
    shell.magic("goto next")
    assert getcwd() == "{0}/this/0".format(tmpdir)
    assert interactive.jobfolder.name == "/this/0/"
    shell.magic("goto next")
    assert getcwd() == "{0}/this/1".format(tmpdir)
    assert interactive.jobfolder.name == "/this/1/"
    shell.magic("goto next")  # no further jobs
    assert getcwd() == "{0}/this/1".format(tmpdir)
    assert interactive.jobfolder.name == "/this/1/"
    shell.magic("goto /")  # go back to root to avoid errors

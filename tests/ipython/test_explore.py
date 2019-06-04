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
from sys import version_info
if version_info[0] == 2:
    from mock import patch
else:
    from unittest.mock import patch


def test(shell, tmpdir, functional):
    from os.path import join
    from pylada.jobfolder import JobFolder
    import pylada

    with patch("pylada.misc.cmdl_input", return_value="y"):

        root = JobFolder()
        for type, trial, size in [
            ("this", 0, 10),
            ("this", 1, 15),
            ("that", 2, 20),
            ("that", 1, 20),
        ]:
            jobfolder = root / type / str(trial)
            jobfolder.functional = functional
            jobfolder.params["indiv"] = size
            if type == "that":
                jobfolder.params["value"] = True

        shell.user_ns["jobfolder"] = root
        shell.magic("explore jobfolder")
        jobfolder = pylada.interactive.jobfolder
        assert (
            "this/0" in jobfolder
            and "this/1" in jobfolder
            and "that/2" in jobfolder
            and "that/1"
        )
        assert "0" in jobfolder["this"] and "1" in jobfolder["this"]
        assert "1" in jobfolder["that"] and "2" in jobfolder["that"]
        assert "other" not in jobfolder
        for job in jobfolder.values():
            assert repr(job.functional) == repr(functional)
        assert getattr(jobfolder["this/0"], "indiv", 0) == 10
        assert getattr(jobfolder["this/1"], "indiv", 0) == 15
        assert getattr(jobfolder["that/1"], "indiv", 0) == 20
        assert getattr(jobfolder["that/2"], "indiv", 0) == 20
        assert not hasattr(jobfolder["this/0"], "value")
        assert not hasattr(jobfolder["this/1"], "value")
        assert getattr(jobfolder["that/1"], "value", False)
        assert getattr(jobfolder["that/2"], "value", False)
        assert pylada.interactive.jobfolder_path is None
        assert "jobparams" in shell.user_ns
        assert jobfolder is shell.user_ns["jobparams"].jobfolder

        shell.magic("savefolders {0}/dict".format(tmpdir))
        pylada.interactive.jobfolder = None
        pylada.interactive.jobfolder_path = None
        shell.magic("explore {0}/dict".format(tmpdir))
        jobfolder = pylada.interactive.jobfolder
        assert (
            "this/0" in jobfolder
            and "this/1" in jobfolder
            and "that/2" in jobfolder
            and "that/1"
        )
        assert "0" in jobfolder["this"] and "1" in jobfolder["this"]
        assert "1" in jobfolder["that"] and "2" in jobfolder["that"]
        assert "other" not in jobfolder
        for job in jobfolder.values():
            assert repr(job.functional) == repr(functional)
        assert getattr(jobfolder["this/0"], "indiv", 0) == 10
        assert getattr(jobfolder["this/1"], "indiv", 0) == 15
        assert getattr(jobfolder["that/1"], "indiv", 0) == 20
        assert getattr(jobfolder["that/2"], "indiv", 0) == 20
        assert not hasattr(jobfolder["this/0"], "value")
        assert not hasattr(jobfolder["this/1"], "value")
        assert getattr(jobfolder["that/1"], "value", False)
        assert getattr(jobfolder["that/2"], "value", False)
        assert pylada.interactive.jobfolder_path is not None
        assert "jobparams" in shell.user_ns
        assert jobfolder is shell.user_ns["jobparams"].jobfolder
        assert jobfolder is shell.user_ns["collect"].jobfolder

        for name, job in root.items():
            if name == "this/1":
                continue
            job.compute(outdir=str(tmpdir.join(name)))

        shell.magic("explore results".format(tmpdir))
        assert {"/this/0/", "/that/1/", "/that/2/"} == set(
            shell.user_ns["collect"].keys()
        )
        shell.magic("explore errors".format(tmpdir))
        assert len(shell.user_ns["collect"]) == 0
        shell.magic("explore all".format(tmpdir))
        shell.magic("explore errors".format(tmpdir))
        assert set(shell.user_ns["collect"].keys()) == {"/this/1/"}



wrapUpload.py
==============

A researcher runs the Python program wrapUpload.py to
upload results to the server.
The program takes as input either ...

  * A directory tree, in which case command line parameters
    control which subdirectories are accepted for uploading.
  * A list of specific directories to upload.

The wrapUpload program creates a unique identifier
for this upload, called the ``wrapId``.
See the `SQL database overview <sqlDatabase.html>`_
for more information on the ``wrapId``.

Then wrapUpload makes a list of all files to be
uploaded.  These must be in directories that passed the
filters above and is restricted to the files:

  * metadata: user-specified metadata like name and publication DOI.
  * INCAR: Main list of parameters.
  * KPOINTS: Specify the k-point grid.
  * POSCAR: Specify atom positions.
  * POTCAR: Specify atomic pseudopotentials.
  * OUTCAR: General log and results in human-readable format.
  * vasprun.xml: All results in XML format.

Then wrapUpload makes a single JSON file, ``wrapId.json``,
containing metadata such as the list of directories.
Then wrapUpload makes a single compressed tar file, ``wrapId.tgz``,
containing all the files.  Finally wrapUpload
uses ``scp`` to upload three files:

  * wrapId.json: General statistics and information
  * wrapId.tgz: Compendium of all files to be archived.
  * wrapId.zzflag: Zero-length flag.

The wrapId.zzflag file gets uploaded last.  The server
process `wrapReceive <wrapReceive.html>`_ doesn't
start processing the files until receiving the flag file,
thereby preventing the server from starting
to process partly-received data.

-------------------------------------------------------

.. automodule:: nrelmat.wrapUpload

.. currentmodule:: nrelmat.wrapUpload
.. autofunction:: main
.. autofunction:: doUpload
.. autofunction:: searchDirs
.. autofunction:: iterateDirs
.. autofunction:: processDir
.. autofunction:: getStatMap
.. autofunction:: getStatInfos
.. autofunction:: getIcsdMap
.. autofunction:: unused_extractPotcar
.. autofunction:: parseMetadata
.. autofunction:: checkFileFull
.. autofunction:: checkFile
.. autofunction:: runSubprocess
.. autofunction:: findNumFiles
.. autofunction:: formatUui
.. autofunction:: parseUui
.. autofunction:: printMap
.. autofunction:: formatMatrix
.. autofunction:: logit
.. autofunction:: throwerr





wrapReceive.py
==============

The server runs a single Python program: wrapReceive.py.
Every few seconds wrapReceive checks for files sent into the 
directory /data/incoming by the client process
`wrapUpload <wrapUpload.html>`_.

In particular, ``wrapReceive`` checks for files having the
format ``wrapId.zzflag``.  
See the `SQL database overview <sqlDatabase.html>`_
for more information on the ``wrapId``.

If ``wrapReceive`` finds a file there having
the format ``wrapId.zzflag``, ...

  * It checks that the three files are present:

    * wrapId.json: General statistics and information
    * wrapId.tgz: Compendium of all files to be archived.
    * wrapId.zzflag: Zero-length flag.

  * It creates a directory /data/arch/wrapId, and moves
    moves the three files from /data/incoming to /data/arch/wrapId.

  * Within directory /data/arch/wrapId ...
  * It untars wrapId.tgz to subdirectory vdir
  * It calls fillDbVasp.py, passing the directory /data/arch/wrapId.

-------------------------------------------------------

.. automodule:: nrelmat.wrapReceive

.. currentmodule:: nrelmat.wrapReceive
.. autofunction:: main
.. autofunction:: gatherArchive
.. autofunction:: processTree
.. autofunction:: checkDupProcs
.. autofunction:: throwerr

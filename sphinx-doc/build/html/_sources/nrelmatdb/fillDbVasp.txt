
fillDbVasp.py
=============

The fillDbVasp program has three possible main functions:

  * Delete and recreate the ``model`` database table.
    This function is rarely used.

  * Delete and recreate the ``contrib`` database table.
    This function is rarely used.

  * Analyze the contents of a directory corresponding to
    a single wrapId, and ingest the data into the database.
    This is the function invoked by wrapReceive.py.

Continuing with the third choice ...

For each relative subdir specified in the wrapId.json file,
fillDbVasp finds the reconstructed directory and calls
fillRow to handle the directory.
The fillRow method calls readVasp.py to read the vasprun.xml file.
The readVasp.py program returns a map (Python dictionary)
and adds a single row to the model table.

Finally, fillDbVasp adds a single row to the contrib
table with information about the wrapId covering the entire
set of directories.

-------------------------------------------------------

.. automodule:: nrelmat.fillDbVasp

.. currentmodule:: nrelmat.fillDbVasp
.. autofunction:: main
.. autofunction:: fillDbVasp
.. autofunction:: createTableModel
.. autofunction:: createTableContrib
.. autofunction:: fillTable
.. autofunction:: fillRow
.. autofunction:: formatArray
.. autofunction:: throwerr

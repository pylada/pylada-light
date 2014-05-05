.. _vasp_extracting_ug:

Extracting results from VASP calculations
=========================================

Exctracting from a single calculation
-------------------------------------


Launching a calculation is not everything. One wants the results. Pylada makes it
easy to interface with those results directly in the python language. Every
functional in Pylada returns an extraction object capable of grepping the
relevant output.

>>> result = vasp(structure)
>>> print result.success
True
>>> print result.eigenvalues * 2
[5, 6, 7] * eV

The above checks that the calculation ran to completion, and then multiplies
the eigenvalues by two. At this point, one could perform any sort of
post-processing, and then automatically launch a subsequent calculation.

.. warning::

   Success means the calculation ran to completion, specifically that the lines
   giving the elapsed time exist in the OUTCAR. It does not mean that the
   results are meaningful.
             

The extraction object can be obtained without rerunning VASP_. There are to
ways to go about this. One is, well, to rerun (but without rerunning, if that
makes sense):

>>> # first time, vasp is executed, presumably.
>>> result = vasp(structure, outdir='mycalc', ispin=1)
>>>
>>> # abort if not successful
>>> assert result.success 
>>>
>>> ... do something ...
>>> 
>>> # second time, OUTCAR exists. It is NOT overwritten.
>>> # The extraction object is returned immediately.
>>> result = vasp(structure, outdir='mycalc', ispin=2) 
 
In the example above, VASP_ is actually launched the first time. However, on the
second pass, an OUTCAR is found. If it is a successful run, then Pylada will
*not* overwrite it. It does not matter whether the structure has changed, or
whether the VASP_ parameters are different. Pylada will *never* overwrite a
successful run. Not unless specifically requested to. The returned extraction
object corresponds to the OUTCAR. Hence, on the second pass, it is the results
of the first call which are returned. Unless, of course, a successful
calculation already existed there prior to the first run, in which case Pylada
would *never ever* have been so crass as to overwrite it.


The second method is to create an extraction object directly:

>>> from pylada.vasp import Extract
>>> result = Extract('/path/to/directory')

In the above, it is expected that the OUTCAR is called OUTCAR. The path can
also be made to a file with a name other than OUTCAR. 

.. note::

   An extraction object can be created for any OUTCAR, whether obtained *via*
   Pylada or not. Some information that Pylada automatically appends to an OUTCAR
   may not be obtainable, however.

To find out what Pylada can extract, do ``result.[TAB]`` in the ipython_
environment. Or checkout :py:class:`~pylada.vasp.extract.Extract`.

If you know how to use `regular expressions`_, creating a property like those above
is generally fairly simple. Edit the file "vasp/extract/base.py", reinstall,
and you're golden. And send your snippet back this way.

.. _vasp_massextract_ug:

Extracting results from *many* calculations, and plotting stuff
---------------------------------------------------------------

Pylada arranges all calculations within directories, with a single VASP
calculation per sub-directory. It can be expedient to extract simultaneously
all the results contained within a directory and its subdirectories. One
approach is to use :ref:`jobfolders <jobfolder_ug>` and the :ref:`ipython
interface <ipython_ug>`. Failing that, however, it still possible to extract
all the results within a tree of directories simultaneously. When used in
conjunction with plotting software such as matplotlib_, it makes it really easy
to synthesize and understand the results from a set of calculations.
It all comes down to a few simple lines:

>>> from pylada.vasp import MassExtract
>>> a = MassExtract('some/path')
>>> a.total_energies
{
   '/this/path/':  array(-666.667) * eV,
   '/that/path/':  array(-999.998) * eV
}

"this/path" and "that/path" are directories in "some/path" where OUTCAR files
exist.  The return is a
:py:class:`~pylada.jobfolder.forwarding_dict.ForwardingDict` instance. It is
possible to string together attributes to get to those of interest:

>>> a.structure.scale
{
   '/this/path/':  5.45, 
   '/that/path/':  5.65
}

From there, it is one simple step to plot, say, energies with respect to the
scale (first, run ipython with the ``-pylab`` flag to import matplotlib_
related stuff, or run the ipython notebook app):

>>> x = array(a.structure.scale.values())
>>> y = array(a.total_energies.values())
>>> plot x, y


:py:class:`~pylada.vasp.extract.MassExtract` behaves exactly like the
:ref:`collect <ipython_collect_ug>` object.

.. _regular expressions: http://docs.python.org/library/re.html

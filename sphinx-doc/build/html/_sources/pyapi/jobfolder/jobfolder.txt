Jobdict class
*************

.. automodule:: pylada.jobfolder.jobfolder
.. moduleauthor:: Mayeul d'Avezac
.. autoclass:: JobFolder
   :show-inheritance:

   .. automethod:: __init__

   .. attribute:: children

      A dictionary holding instances of subfolders further down the tree.

   .. attribute:: params

      A dictionary holding the parameters for the calculation in this
      particular folder, if any. The parameters will be passed on to the
      functional as ``**params``.

   .. attribute:: parent

      Reference to the :py:class:`JobFolder` instance which holds this
      one, i.e. the parent directory. If None, then this folder-dictionary is
      the root.

      .. note:: This back-reference creates reference cycles. However, we
         cannot use a weak-reference here, unless we are OK with risking the
         loss of a parent folder when only a reference to the subfolder is held
         by the user.

   .. autoattribute:: functional
   .. autoattribute:: name
   .. autoattribute:: is_executable
   .. autoattribute:: untagged_folders
   .. autoattribute:: is_tagged
   .. autoattribute:: nbfolders
   .. autoattribute:: subfolders
   .. autoattribute:: root

   .. automethod:: __iter__
   .. automethod:: iteritems
   .. automethod:: itervalues
   .. automethod:: iterkeys
   .. automethod:: items
   .. automethod:: values
   .. automethod:: keys
   .. automethod:: subfolders()->[str, str,...]

   .. automethod:: update(JobFolder)->None

   .. automethod:: compute

   .. automethod:: tag()->None
   .. automethod:: untag()->None

   .. automethod:: __getattr__
   .. automethod:: __setattr__
   .. automethod:: __div__
   .. automethod:: __getitem__
   .. automethod:: __setitem__
   .. automethod:: __contains__

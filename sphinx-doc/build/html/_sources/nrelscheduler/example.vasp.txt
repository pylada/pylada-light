
.. _example.vasp:

schedMain Example B: VASP
====================================


.. graphviz::


    digraph exampleStatic {
      //graph [label="NREL MatDB Data Flow", labelloc=t, fontsize=30];
      rank=source;
      legendx [shape=none, margin=0, label=<
        <table border="0" cellborder="0" cellspacing="0" cellpadding="1">
        <tr><td><font point-size="20"><b> Task Dependencies </b></font></td></tr>
        <tr><td><font point-size="12">
        All of the following tasks run in directory<br/>
        icsd_06845 or a subdirectory of it.<br/>
        However there could be many icsd_*<br/>
        directories specified in the initWork file,<br/>
        and all would run in parallel.
        </font></td></tr>
        </table>
      >];
      rankdir = TB;
      node [color=blue, shape=box, fontsize=10];
      edge [fontsize=11];
      //URL="index.html";       // default for entire graph


      nonmagSetup [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            nonmagSetup.py in dir icsd_06845
          </u></b></font> </td></tr>
          <tr> <td align="left">  * Specified in initWork </td></tr>
          <tr> <td align="left">  * Starts immediately (no prereqs) </td></tr>
          <tr><td> <b> Sets up the next step by: </b> </td></tr>
          <tr> <td align="left">  * Writes nonmagSetup.postOkWork, which specifies runVaspChain.py in dir non-magnetic </td></tr>
          <tr> <td align="left">  * Writes non-magnetic/runVaspChain.preWork, which specifies prerequisite nonmagSetup.py </td></tr>
          <tr><td> <b> Sets up the step after runVaspChain by: </b> </td></tr>
          <tr> <td align="left">  * Writes non-magnetic/runVaspChain.postOkWork, which specifies magSetup.py </td></tr>
          <tr> <td align="left">  * Writes magSetup.preWork, which specifies prerequisite runVaspChain </td></tr>
        </table>
      >];


      nonmagVasp [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            runVaspChain.py in dir non-magnetic
          </u></b></font> </td></tr>
          <tr> <td align="left"> * In subdir relax_cellShape, runs VASP cell shape relaxation  </td></tr>
          <tr> <td align="left"> * In subdir relax_ions, runs VASP cell ionic relaxation  </td></tr>
          <tr> <td align="left"> * In subdir final, runs one final VASP </td></tr>
        </table>
      >];


      magSetup [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            magSetup.py in dir icsd_06845
          </u></b></font> </td></tr>
          <tr> <td align="left">  * Starts after non-mag runVaspChain </td></tr>
          <tr><td> <b> Sets up the next tasks by: </b> </td></tr>
          <tr> <td align="left">  * Writes magSetup.postOkWork, which specifies runVaspChain.py in all the mag dirs: ls-ferro, hs-ferro, and *anti-ferro* for larger structures </td></tr>
          <tr> <td align="left">  * Writes runVaspChain.preWork in all the mag dirs, which specifies prerequisite magSetup.py </td></tr>
          <tr><td> <b> Sets up the step after all the runVaspChains complete by: </b> </td></tr>
          <tr> <td align="left">  * Writes runVaspChain.postOkWork in all the mag dirs, which specifies gwSetup.py </td></tr>
          <tr> <td align="left">  * Writes gwSetup.preWork, which specifies prerequisites runVaspChain in all the mag dirs </td></tr>
        </table>
      >];




      magVasp [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            runVaspChain.py (multiple tasks, one in each of the mag dirs)
          </u></b></font> </td></tr>
          <tr> <td align="left"> * In subdir relax_cellShape, runs VASP cell shape relaxation  </td></tr>
          <tr> <td align="left"> * In subdir relax_ions, runs VASP cell ionic relaxation  </td></tr>
          <tr> <td align="left"> * In subdir final, runs one final VASP </td></tr>
        </table>
      >];


      gwSetup [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            gwSetup.py in dir icsd_06845
          </u></b></font> </td></tr>
          <tr> <td align="left">  * Starts after all the mag runVaspChains </td></tr>
          <tr><td> <b> Sets up the next step by: </b> </td></tr>
          <tr> <td align="left">  * Writes gwSetup.postOkWork, which specifies runVaspChain.py in dir gwdir </td></tr>
          <tr> <td align="left">  * Writes gwdir/runVaspChain.preWork, which specifies prerequisite gwSetup.py </td></tr>
        </table>
      >];



      gwVasp [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <font point-size="12"><b><u>
            runVaspChain.py in dir gwdir
          </u></b></font> </td></tr>
          <tr> <td align="left"> * In subdir relax_cellShape, runs VASP cell shape relaxation  </td></tr>
          <tr> <td align="left"> * In subdir relax_ions, runs VASP cell ionic relaxation  </td></tr>
          <tr> <td align="left"> * In subdir final, runs one final VASP </td></tr>
        </table>
      >];



      legendx -> nonmagSetup [style=invis];
      nonmagSetup -> nonmagVasp
      nonmagVasp -> magSetup
      magSetup -> magVasp
      magVasp -> gwSetup
      gwSetup -> gwVasp
    }


Demo mode
---------

We will run this example twice - first in demo mode,
without invoking VASP, and later with VASP.
Here in demo mode we will make two abreviations:

  * Instead of using the HPC queueing system with ``qsub``,
    we will execute scripts on the local host with bash.
  * Instead of calling VASP we will simply copy in the results
    of a previous VASP run.  The results aren't meaningful,
    but it allows us to demonstrate quickly the overall schedMain framework.

To run this example, pick a name of some directory
for testing, say ``testb``.  Then::

  cp -r schedMain/example.vasp testb
  cd testb

  # Set up dummy files to be used instead of calling VASP
  cp -r demoFiles global/vaspDemo

Make sure your PYTHONPATH includes ``nrelmat/readVasp.py``.
For example::

  export PYTHONPATH=$PYTHONPATH:.../nrelmat

Finally, run the scheduler::

  .../schedMain.py -globalDir global -ancDir . -initWork initWork -delaySec 1 -redoAll n



The output format and task status values are documented
in :ref:`example.static`.


Real mode
+++++++++++

Real mode means we will use the HPC queueing system and we
will execute VASP.  This will run **much** more slowly
than our quick demo.  First we need to inform the scheduler
about the HPC queueing system and our VASP.

Set up the HPC queueing system
---------------------------------

For example assume your system is named "myHpc".
In the file ``schedMisc.py`` you will find a section like::

  if hostType == 'peregrine': cmdLine = 'showq'

You need to add a similar section that specifies the command
used on myHPC to show the queue; something like::

  elif hostType == 'myHpc': cmdLine = 'showq'

In schedMisc.py you also will find a section like::

    if hostType == 'peregrine':
      if len(qtoks) == 9 and re.match('^\d+$', qtoks[0]):   # ignore headings
        (qjobId, quserId, qstate) = qtoks[0:3]
        if qstate in ['BatchHold', 'Hold', 'Idle', 'SystemHold', 'UserHold']:
          status = ST_WAIT
        elif qstate in ['Running']: status = ST_RUN
        else:
          print 'getPbsMap: unknown status: %s' % (qline,)
          status = ST_WAIT              # assume it's some variant of wait
        pbsMap[qjobId] = status

You need to add a similar section for myHPC.  The new
section must extract from the showq output the job ID and
the job status, and must translate the status to one of the
ST_* constants at the top of schedMisc.py.

Similarly, search for ``hostType`` and add sections appropriate
for myHpc in the following files::

  * taskClass.py
  * example.vasp/global/cmd/magSetup.py
  * example.vasp/global/cmd/nonmagSetup.py
  * example.vasp/global/cmd/runVaspChain.py
  * example.vasp/global/cmd/rvmisc.py








Set up VASP
-------------

In your home directory create a file named ``pyladaExec.sh``,
and use an ascii text editor (vim, emacs, etc) to add
content like the following::

  # Add any setup you like, such as export PATH,
  # export LD_LIBRARY_PATH, etc.

  # Start VASP
  .../my/path/to/VASP

Make sure the file is executable::

  chmod u+x pyladaExec.sh

Delete the dummy VASP files we used in the demo above::

  rm -r global/vaspDemo


Run schedMain
------------------

Finally we run the scheduler::

  .../schedMain.py -globalDir global -ancDir . -initWork initWork -delaySec 5 -redoAll n -hostType myHpc

This could run for a few minutes to a few days, depending on
your HPC queues.  Using ``-delaySec 5`` instead of 1 makes the
display less busy, but the job runs essentially as fast.



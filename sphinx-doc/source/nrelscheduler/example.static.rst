
.. _example.static:

schedMain Example A: static files
====================================


.. graphviz::


    digraph exampleStatic {
      //graph [label="NREL MatDB Data Flow", labelloc=t, fontsize=30];
      rank=source;
      legendx [shape=none, margin=0, label=<
        <table border="0" cellborder="0" cellspacing="0" cellpadding="1">
        <tr><td><font point-size="20"> Task Dependencies </font></td></tr>
        </table>
      >];
      rankdir = TB;
      node [color=blue, shape=box, fontsize=10];
      edge [fontsize=11];
      //URL="index.html";       // default for entire graph


      alpha0 [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> alpha.sh in aaDir </u></b> </td></tr>
          <tr> <td align="left">  * Specified in initWork </td></tr>
          <tr> <td align="left">  * Starts immediately (no prereqs) </td></tr>
          <tr> <td align="left">  * At end writes file aaDir/alpha.status.ok </td></tr>
        </table>
      >];


      beta0 [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center">  <b><u> beta.py in bbDir0 </u></b> </td></tr>
          <tr> <td align="left"> * Specified in aaDir/alpha.postOkWork </td></tr>
          <tr> <td align="left"> * Starts after alpha.sh ends </td></tr>
          <tr> <td align="left"> * At end writes file bbDir0/beta.status.ok </td></tr>
        </table>
      >];




      beta1 [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> beta.py in bbDir1 </u></b> </td></tr>
          <tr> <td align="left"> * Specified in aaDir/alpha.postOkWork </td></tr>
          <tr> <td align="left"> * Starts after alpha.sh ends </td></tr>
          <tr> <td align="left"> * At end writes file bbDir1/beta.status.ok </td></tr>
        </table>
      >];



      beta2 [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> beta.py in bbDir2 </u></b> </td></tr>
          <tr> <td align="left"> * Specified in aaDir/alpha.postOkWork </td></tr>
          <tr> <td align="left"> * Starts after alpha.sh ends </td></tr>
          <tr> <td align="left"> * At end writes file bbDir2/beta.status.ok </td></tr>
        </table>
      >];


      gamma0 [shape=none, margin=0, label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> gamma.py in ccDir </u></b> </td></tr>
          <tr> <td align="left"> * Specified in aaDir/alpha.postOkWork </td></tr>
          <tr> <td align="left"> * Starts after all 3 beta.py end </td></tr>
          <tr> <td align="left"> * At end writes file ccDir/gamma.status.ok </td></tr>
        </table>
      >];

      legendx -> alpha0 [style=invis];
      alpha0 -> beta0
      alpha0 -> beta1
      alpha0 -> beta2
      beta0 -> gamma0
      beta1 -> gamma0
      beta2 -> gamma0
    }



To run this example, pick a name of some directory
for testing, say testa.  Then::

  cp -r schedMain/example.static testa
  cd testa

  .../schedMain.py -globalDir global -ancDir . -initWork initWork -delaySec 1 -redoAll n


The command line parameters are:

  =============  ========  ===================================================
  -bugLev        <string>  Debug level.  Typically 0, 1, or 5.
  -hostType      <string>  System type: hostLocal or peregrine or ...
  -globalDir     <string>  Dir containing global info, including subdir cmd
  -ancDir        <string>  An ancestor dir of all dirs to be processed
  -initWork      <string>  File containing the initial work list
  -delaySec      <string>  Schedule loop delay, seconds
  -redoAll       <bool>    n/y: on restart, redo all even if prior run was ok
  -useReadOnly   <bool>    n/y: only print status; do not start tasks
  =============  ========  ===================================================


The possible task status values are:

  **init** The task is on the work list but not yet started.
  Either it was just added to the work list, and soon will start,
  or it has unsatisfied prerequisites.

  **submit** The task has been submitted to the HPC
  via qsub, msub, or similar, but has not yet been recognized
  by the HPC.

  **wait** The task has been submitted to the HPC
  via qsub, msub, or similar, but has not yet started.

  **start** The task has started

  **ok** The task finished successfully and wrote the file
  taskName.status.ok.

  **error** The task finished but with an error.  Generally
  the error message is in file taskName.status.error.

In this example you should see output like the following.
The "#" notes are mine, after the fact::

  # This is the initial work list.  Here schedMain has
  # just read the file initWork.

  schedMain
  task counts:   init:1
    execName              jobId     new  status  npre  taskDir
    --------              -----     ---  ------  ----  -------
    alpha.sh              None      new  init    *  0  aaDir

  scheduleTasks: start task: alpha.sh         taskDir: aaDir

  # After alpha.sh completes, the work list is as follows.
  # SchedMain noticed the file alpha.status.ok, read alpha.postOkWork,
  # and added the new tasks to the work list.
  # The "npre" column is the number of unsatisfied prerequisites.
  # Here gamma cannot start until the 3 betas complete.
  # The "*" indicates that task is ready to start.

  schedMain
  task counts:   init:4  ok:1
    execName              jobId     new  status  npre  taskDir
    --------              -----     ---  ------  ----  -------
    alpha.sh              None      new  ok         0  aaDir
    beta.py               None      new  init    *  0  bbDir0
    beta.py               None      new  init    *  0  bbDir1
    beta.py               None      new  init    *  0  bbDir2
    gamma.py              None      new  init       3  ccDir

  # Schedmain starts all the ready tasks -- the three betas.
  # Gamma cannot start yet since its prerequisites, in gamma.preWork,
  # are the betas.

  scheduleTasks: start task: beta.py          taskDir: bbDir0
  scheduleTasks: start task: beta.py          taskDir: bbDir1
  scheduleTasks: start task: beta.py          taskDir: bbDir2

  # As soon as the betas start they finish,
  # and finally gamma's pre-requisites are satisfied.

  scheduleTasks: start task: gamma.py         taskDir: ccDir

  # All done.
  # If schedMain ends and some tasks have "init" status,
  # most likely it's because their prerequistites aren't
  # satisfied -- perhaps some prior task failed.

  schedMain
  task counts:   ok:5
    execName              jobId     new  status  npre  taskDir
    --------              -----     ---  ------  ----  -------
    alpha.sh              None      new  ok         0  aaDir
    beta.py               None      new  ok         0  bbDir0
    beta.py               None      new  ok         0  bbDir1
    beta.py               None      new  ok         0  bbDir2
    gamma.py              None      new  ok         0  ccDir

The "new" notation means that the task actually ran.
If you start the scheduler again in this directory, with
the same command as above, you will see::

  schedMain
  task counts:   ok:5
    execName              jobId     new  status  npre  taskDir
    --------              -----     ---  ------  ----  -------
    alpha.sh              None           ok         0  aaDir
    beta.py               None           ok         0  bbDir0
    beta.py               None           ok         0  bbDir1
    beta.py               None           ok         0  bbDir2
    gamma.py              None           ok         0  ccDir

Notice the lack of "new" flags.  The scheduler found the
x.status.ok file for each of the tasks and concluded the
task did not need to be rerun.

If you want to force the scheduler to rerun all tasks even
if they completed OK, specify the command line flag ``-redoAll y``.



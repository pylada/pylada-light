.. _side_by_side_mpi_ug:

Running different MPI calculations side-by-side
-----------------------------------------------

.. currentmodule:: pylada

It is often possible to run calculations side-by-side. One can request 64
processors from a supercomputer and run two VASP_ calculations
simultaneously in the same PBS job. There are a fair number of steps to get this part of Pylada running: 

  1. Set up Pylada to run a single MPI programming as described above
  2. Set the environment variable :py:data:`do_multiple_mpi_programs` to
     True.
  3. Set up :py:data:`figure_out_machines`. This is a string which contains
     a small python script. Pylada runs this python script at the start of
     a PBS/Slurm job to figure out the hostnames of each machine allocated
     to the job. For each *core*, the script should print out a line
     starting with "PYLADA MACHINE HOSTNAME". It will be launched as an MPI
     program on all available cores. By default, it is the following simple
     program:

     .. code-block:: python

         from socket import gethostname
         from mpi import gather, world
         hostname = gethostname()
         results = gather(world, hostname, 0)
         if world.rank == 0:
           for hostname in results:
             print "PYLADA MACHINE HOSTNAME:", hostname
         world.barrier()

     .. note::
       
        This is one of two places where boost.mpi is used. By replacing
        this function with, say, call mpi4py methods, one could remove the
        boost.mpi in Pylada for most use cases. 

     It is important that this function prints out to the standard output
     one line per core (not one line per machine).

     The script is launched by
     :py:meth:`pylada.process.mpi.create_global_comm`.
  4. The names of the machines determined in the previous step are stored
     in :py:data:`default_comm`'s
     :py:attr:`pylada.process.mpi.Communicator.machines` attribute. This is
     simply a dictionary mapping the hostnames determined previously to the
     number of cores. It is possible, however, to modify
     :py:data:`default_comm` after the :py:data:`figure_out_machines`
     script is launched and the results parsed. This is done via the method
     :py:meth:`modify_global_comm`. This method takes a
     :py:class:`pylada.process.mpi.Communicator` instance on input and
     modifes it in-place. By default, this method does nothing.

     On a cray, one could set it up as follows:

     .. code-block:: python

        def modify_global_comm(comm):
          """ Modifies global communicator to work on cray.
      
              Replaces hostnames with the host number. 
          """ 
          for key, value in comm.machines.items():
            del comm.machines[key]
            comm.machines[str(int(key[3:]))] = value

     This would replace the hostnames with something aprun can use for MPI
     placement. :py:meth:`modify_global_comm` is runned once at the
     beginning of a Pylada PBS/Slurm script.

  5. To test that the hostnames where determined correctly, one should copy
     the file "process/tests/globalcomm.py" somewhere, edit it, and launch
     it. The names of the machines should be printed out correctly, with
     the right number of cores:

     .. code-block:: bash

       > cd testhere
       > cp /path/to/pylada/source/process/tests/globalcomm.py
       > vi globalcomm.py
       # This is a PBS script.
       # Modify it so it can be launched.
       > qsub globalcomm.py
       # Then, when it finishes:
       > cat global_comm_out
       EXPECTED N=64 PPN=32
       FOUND
       n 64
       ppn 32
       placement ""
       MACHINES
       PYLADA MACHINE HOSTNAME hector.006 32
       PYLADA MACHINE HOSTNAME hector.006 32
       ...


     The above is an example output. One should try and launch this routine
     on more than one node, with varying number of processes per node, and
     so forth.
 

   6. At this point, Pylada knows the name of each machine participating in
      a PBS/Slurm job. It still needs to be told how to run an MPI job on a
      *subset* of these machines. This will depend on the actual MPI
      implementation installed on the machine. Please first read the manual
      for your machine's MPI implementation.

      Pylada takes care of MPI placements by formatting the
      :py:data:`mpirun_exe` string adequately. For this reason, it is
      expected that :py:data:`mpirun_exe` contains a "{placement}" tag
      which will be replaced with the correct value at runtime. 

      At runtime, before placing the call to an external MPI program, the
      method :py:meth:`pylada.machine_dependent_call_modifier` is called.
      It takes three arguments: a dictionary with which to format the
      :py:data:`mpirun_exe` string, a dictionary or
      :py:data:`pylada.process.mpi.Communicator` instance containing
      information relating to MPI, a dictionary containing the environment
      variables in which to run the MPI program. The first and second
      dictionary will be merged and used to format the
      :py:data:`mpirun_exe` string. By default, this method creates a
      nodefile with only those machines involved in the current job. It
      then sets "placement" to "-machinefile filename" where filename is
      the nodefile. 

      On Crays, one could use the following:

      .. code-block:: python 

         def machine_dependent_call_modifier( formatter=None, 
                                              comm=None, 
                                              env=None ):
           """ Placement modifications for aprun MPI processes. 
              
               aprun expects the machines (not cores) to be given on the
               commandline as a list of "-Ln" with n the machine number.
               """
           from pylada import default_comm
           if formatter is None: return
           if len(getattr(comm, 'machines', [])) == 0: placement = ""
           elif sum(comm.machines.itervalues()) == sum(default_comm.machines.itervalues()):
             placement = ""
           else:
             l = [m for m, v in comm.machines.iteritems() if v > 0]
             placement = "-L{0}".format(','.join(l))
           formatter['placement'] = placement

     Note that the above requires the :py:meth:`pylada.modify_global_comm`
     from point 4.
     
     .. warning::

        All external program calls are routed through this function,
        whether or not it is an MPI program. Hence it is necessary to check
        that the program is to be launched an MPI or not. In the case of
        serial programs, "comm" may be None.

  6. The whole deal can be tested using "process/tests/placement.py"
     This is a PBS job which performs MPI placement on a fake job.
     It should be copied somewhere, edited, and launched. 

     At least two arguments should be set prior to running this script.
     Check the bottom of the script. "ppn" specifies the number of
     processors per nodes. The job should be launched with 2*"ppn" cores.
     "path" should point to the source directory of Pylada. This is so that
     a small program can be found (pifunc) and used for testing. The
     program can be compiled by Pylada by setting "compile_test True" in
     cmake_.

     "placement.py" will launch several *simultaneous* instances of the
     "pifunc" program: one long on three quarters of allocated cores, and
     two smaller calculations on one eigth of the cores each. 

     One should check the ouput to make sure that the programs are running
     side-by-side (not all piled up on the same node), that they are
     runnning simultaneously, and that they run successfully (e.g. mpirun
     does launch them).

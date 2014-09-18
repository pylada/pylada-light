

====================================
NREL MatDB SQL Database Overview
====================================

The SQL database consists of two tables, model and contrib.
The column wrapId is used to link them.

The wrapId column
=====================

A wrapId uniquely identifies a single invocation of wrapUpload.py
to upload a group of directories.
A typical wrapId is: ::

    @2013.08.13@12.58.22.735311@someUser@home.someUser.redmesa.old.td.testlada.2013.04.06.Fe.O@

The wrapId is broken into fields separated by "@":

  * Date, yyyy.mm.dd
  * Time, hh.mm.ss.uuuuuu  where uuuuuu is microseconds padded to 6 characters
  * User name.
  * Top directory specified to wrapUpload, with "/" replaced by ".".


The model table
=================

The model table has an auto-increment key, mident.
The mident value is used as a unique indentifier for entries
in the MatDB.  The model table has about 60 columns,
and the columns fall into a few main categories:

General metadata columns
--------------------------

Some example general metadata columns are:

   ============    ========   =============================================
   Column          SQL type   Description
   ============    ========   =============================================
   mident          serial     Unique auto-increment ID
   wrapid          text       Upload identifier, link with contrib table
   abspath         text       Absolute path to dir
   relpath         text       Relative path below topDir
   ============    ========   =============================================


ICSD info columns
--------------------------

The ICSD info is present only on runs adhering
to the ICSD file naming conventions.
To extract ICSD info from the file names, file names must be like: ::

    .../icsd_083665/icsd_083665.cif/ls-anti-ferro-7/relax_cellshape/1
             ^^^^^^                 ^^^^^^^^      ^ ^^^^^^^^^^^^^^^ ^
            icsdNum                 magType  magNum relaxType       relaxNum

Some example ICSD information columns are:

  ============    ========   =========================================
  Column          SQL type   Description
  ============    ========   =========================================
  icsdNum         int        ICSD number in CIF file
  magType         text       type of magnetic moment hs-ferro, etc.
  magNum          int        number of hs-anti-ferro or ls-anti-ferro
  relaxType       text       Type of run: relax-cellshape, etc.
  relaxNum        int        Folder num for relax runs
  ============    ========   =========================================




VASP input columns
--------------------------

Some example VASP input columns are:

  ============    ========   =========================================
  Column          SQL type   Description
  ============    ========   =========================================
  encut_ev        double     VASP ENCUT parameter from INCAR
  ibrion          int        VASP IBRION parameter from INCAR
  isif            int        VASP ISIF parameter from INCAR
  ============    ========   =========================================


VASP output columns
--------------------------

The majority of the columns in the model table are VASP
output values.
Some example VASP output columns are:

  ======================  =====================   ====================
  Column                  SQL type                Description
  ======================  =====================   ====================
  typeNames               text[]                  ['Mo', 'S']
  typeNums                int[]                   [2, 4]
  finalBasisMat           double precision[][]
  finalRecipBasisMat      double precision[][]
  finalForceMat_ev_ang    double precision[][]    eV/angstrom
  finalStressMat_kbar     double precision[][]    kbar
  finalPressure_kbar      double precision        kbar
  eigenMat                double precision[][]
  energyNoEntrp           double precision        eV
  energyPerAtom           double precision        eV
  ======================  =====================   ====================


Author metadata columns
--------------------------

The author metadata columns derive from the ``metadata`` files
specified by the researcher.
Some example author metadata columns are:

  ==================  ========  ==========================================
  Column              SQL type  Description
  ==================  ========  ==========================================
  hashstring          text      sha512 of our vasprun.xml
  meta_parents        text[]    sha512 of parent vasprun.xml, or null
  meta_firstName      text      metadata: first name
  meta_lastName       text      metadata: last name
  meta_publications   text[]    metadata: publication DOI or placeholder
  meta_standards      text[]    metadata: controlled vocab keywords
  meta_keywords       text[]    metadata: uncontrolled vocab keywords
  meta_notes          text      metadata: notes
  ==================  ========  ==========================================


The contrib table
=================

The contrib table is brief, consisting of ...

  ================  ===========  =============================================
  Column            SQL type     Description
  ================  ===========  =============================================
  wrapid            text         wrapId for this upload, link with model table
  curdate           timestamp    date, time of this wrapId
  userid            text         user id doing the upload
  hostname          text         hostname of the upload
  topDir            text         top level dir of the upload
  numkeptdir        int          num of subdirs uploaded
  reldirs           text[]       list of relative subdirs
  ================  ===========  =============================================



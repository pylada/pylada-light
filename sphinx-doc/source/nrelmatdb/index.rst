.. nrelmat documentation master file, created by
   sphinx-quickstart on Fri May 17 13:45:57 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. graphviz::


    digraph nrelmat {
      //graph [label="NREL MatDB Data Flow", labelloc=t, fontsize=30];
      rank=source;
      legendx [shape=none, margin=0, label=<
        <table border="0" cellborder="0" cellspacing="0" cellpadding="1">
        <tr><td><font point-size="30"> NREL MatDB Data Flow </font></td></tr>
        <tr><td><font point-size="20"> Click any box for details </font></td></tr>
        </table>
      >];
      rankdir = TB;
      node [color=blue, shape=box, fontsize=11];
      edge [fontsize=11];
      //URL="index.html";       // default for entire graph

      wrapUpload [URL="wrapUpload.html", shape=none, margin=0,
        tooltip="Click for wrapUpload details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> wrapUpload </u></b> </td> </tr>
          <tr> <td align="left"> Find candidate directories </td> </tr>
          <tr> <td align="left"> Gather metadata </td> </tr>
          <tr> <td align="left"> Use <font face="courier-bold">tar</font> to create an archive file </td> </tr>
          <tr> <td align="left"> Use <font face="courier-bold">scp</font> to copy the file to the archive server </td> </tr>
        </table>
      >];

      fillDbVasp [URL="fillDbVasp.html", shape=none, margin=0,
        tooltip="Click for fillDbVasp details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> fillDbVasp </u></b> </td> </tr>
          <tr> <td align="left"> Traverse directory tree </td> </tr>
          <tr> <td align="left"> For each dir: </td> </tr>
          <tr> <td align="left">       Call readVasp to extract statistics </td> </tr>
          <tr> <td align="left">       Add a row to the model table </td> </tr>
          <tr> <td align="left"> Call augmentDb to calc additional statistics for the model table </td> </tr>
          <tr> <td align="left"> Add a row to the contrib table </td> </tr>
        </table>
      >];

      augmentDb [URL="augmentDb.html", shape=none, margin=0,
        tooltip="Click for augmentDb details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> augmentDb </u></b> </td> </tr>
          <tr> <td align="left"> Create derived fields for SQL database </td> </tr>
        </table>
      >];

      flatDatabase [URL="flatDatabase.html", shape=none, margin=0,
        tooltip="Click for flat file database details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> Flat file database </u></b> </td> </tr>
          <tr> <td> Contains compressed copies<br/>of all uploaded files </td> </tr>
        </table>
      >];


      sqlDatabase [URL="sqlDatabase.html", shape=none, margin=0,
        tooltip="Click for PostgreSQL database details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td align="center"> <b><u> PostgreSQL database </u></b> </td> </tr>

          <tr> <td align="left"> model table: Summary statistics, one row per vasp run </td> </tr>
          <tr> <td align="left"> contrib table: Author statistics, one row per upload </td> </tr>
        </table>
      >];

      webServer [URL="webServer.html", shape=none, margin=0,
        tooltip="Click for web server details", label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="1">
          <tr> <td colspan="2" align="center"> <b><u> Web server </u></b> </td> </tr>
          <tr> <td align="left"> Language: </td> <td align="left"> Python</td> </tr>
          <tr> <td align="left"> Platform: </td> <td align="left"> Pyramid</td> </tr>
          <tr> <td align="left"> Template system: </td> <td align="left"> Mako</td> </tr>
          <tr> <td align="left"> 3-D Visualization: </td> <td align="left"> WebGL using three.js</td> </tr>
        </table>
      >];

      legendx -> wrapUpload [style=invis];
      wrapUpload -> fillDbVasp
      fillDbVasp -> flatDatabase
      fillDbVasp -> sqlDatabase
      sqlDatabase -> augmentDb -> sqlDatabase
      sqlDatabase -> webServer
      flatDatabase -> webServer
    }


.. toctree::
   :maxdepth: 1

   SQL database overview: Overview of the database system. <sqlDatabase>
   Flat file database overview: Overview of the database system. <flatDatabase>
   Web overview: Overview of the web server system. <webServer>
   augmentDb.py: Add additional info to the model table. <augmentDb>
   execMimic.py: Mimic execution for validity testing. <execMimic>
   fillDbVasp.py: Read files created by wrapUpload and add rows to the model table.  <fillDbVasp>
   queryMatDb.py: Demo query of the NRELMatDB SQL database. <queryMatDb>
   readVasp.py: Read and parse an OUTCAR or vasprun.xml file <readVasp>
   ScanOutcar.py: Extract info from OUTCAR and INCAR files <ScanOutcar>
   ScanXml.py: Read and parse a vasprun.xml file <ScanXml>
   wrapReceive.py: Receive results sent by wrapUpload.sh <wrapReceive>
   wrapUpload.py: Locate, extract, and upload results to the server running wrapReceive. <wrapUpload>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



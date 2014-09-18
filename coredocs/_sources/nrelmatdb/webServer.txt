

====================================
NREL MatDB Web Server Overview
====================================


The NREL MatDB web server is based on the
Pyramid_ Python web framework, and uses
the Mako_ templating system.

.. _Pyramid: http://www.pylonsproject.org/
.. _Mako: http://www.makotemplates.org/

It requires a customized Python, so there is a virtualenv
Python in virtmako.  The virtualenv was constructed by: ::

  virtualenv virtmako
  . virtmako/bin/activate
  pip install pyramid
  pip install pyramid_beaker
  pip install waitress
  pip install psycopg2

The web app uses a standard Pyramid structure:

  * Within directory TestMako ...
  * Overall definitions are found in development.ini and production.ini
  * Within directory TestMako/testmako ...
  * Routing info is in __init__.py
  * The views (control logic) are defined in views.py.
    Each view calls a corresponding Mako template file
    in the templates directory.  All the Mako templates
    use template inheritance and are based on tmBase.mak.

    =====================     =========================================
    views.py method           template file
    =====================     =========================================
    vwLogin                   tmLogin.mak
    vwLogout                  None; redirects to vwHome
    vwHome                    tmHome.mak
    vwNotFound                tmNotFound.mak
    vwQueryStd                tmQueryStd.mak
    vwQueryAdv                tmQueryAdv.mak
    vwDetail                  tmDetail.mak
    vwDownload                None; produces mime type text/plain.
    vwVisualize               tmVisualize.mak
    =====================     =========================================


###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
# 
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
# 
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
# 
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
# 
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

#!/usr/bin/env python

# Input:
#   lattice
#   incar dict?
#   potenDir
# Output at outDir:
#   4 (or more? ) files

# The cifMap represents the raw cif file.
# The icsdMap has a few simple computed values,
#   and is retrieved by getIcsdMap, which calls mkIcsdMap.
# The vaspMap has more complex computed values,
#   and is retrieved by getVaspMap, which calls mkVaspMap.


import datetime, os, re, shlex, sys, traceback
import numpy as np


class CifException( Exception):

  def __init__( self, val):
    self.val = val

  def __str__( self):
    return repr( self.val)



#====================================================================

def badparms( msg):
  print '\nError: %s' % (msg,)
  print 'Parms:'
  print ''
  print '  Specify either -inFile or (inDir and updateLog).'
  print '  -buglev      <int>     debug level'
  print '  -inFile      <string>  input cif file'
  print ''
  print '  -inDir       <string>  dir containing cif files'
  print '  -updateLog   <string>  updated log of cif files processed'
  print ''
  print '  -inPotenDir  <string>  input dir containing pseudopotential subdirs'
  print '                           Or none.'
  print '  -outPoscar  <string>   output POSCAR file'
  print '                           Or none.'
  print '  -outPotcar  <string>   output POTCAR file'
  print '                           Or none.'
  print ''
  print 'Example:'
  print './readCif.py -buglev 5 -inFile icsd_027856.cif -inPotenDir  ~/vladan/td.pseudos/pseudos -outPoscar temp.poscar -outPotcar temp.potcar | less'
  sys.exit(1)


#====================================================================


#xxx all xxx

#====================================================================


def main():
  buglev = None
  inFile = None
  inDir = None
  updateLog = None
  inPotenDir = None
  outPoscar = None
  outPotcar = None

  if len(sys.argv) % 2 != 1:
    badparms('Parms must be key/value pairs')
  for iarg in range( 1, len(sys.argv), 2):
    key = sys.argv[iarg]
    val = sys.argv[iarg+1]
    if key == '-buglev': buglev = int( val)
    elif key == '-inFile': inFile = val
    elif key == '-inDir': inDir = val
    elif key == '-updateLog': updateLog = val
    elif key == '-inPotenDir': inPotenDir = val
    elif key == '-outPoscar': outPoscar = val
    elif key == '-outPotcar': outPotcar = val
    else: badparms('unknown key: "%s"' % (key,))

  if buglev == None: badparms('parm not specified: -buglev')
  if inFile != None:
    if inDir != None or updateLog != None:
      badparms('May specify either -inFile or (inDir and updateLog)')
  else:
    if inDir == None or updateLog == None:
      badparms('May specify either -inFile or (inDir and updateLog)')
  if inPotenDir == None: badparms('parm not specified: -inPotenDir')
  if outPoscar == None: badparms('parm not specified: -outPoscar')
  if outPotcar == None: badparms('parm not specified: -outPotcar')




  if inFile != None:
    cifRdr = CifReader( buglev, typeMap, inFile)
    icsdMap = cifRdr.getIcsdMap()
    vaspMap = cifRdr.getVaspMap()

    if outPoscar != 'none':
      writePoscar( inFile, vaspMap, outPoscar)
    if inPotenDir != 'none' and outPotcar != 'none':
      writePotcar( vaspMap, inPotenDir, outPotcar)

  else:
    doneNames = []
    fin = open( updateLog)
    while True:
      line = fin.readline()
      if len(line) == 0: break
      line = line.strip()
      if len(line) > 0 and not line.startswith('#'): doneNames.append( line)
    fin.close()
    if buglev >= 2: print 'main: doneNames: %s' % (doneNames,)
    doTree( buglev, doneNames, inDir, updateLog)

#====================================================================

def doTree( buglev, doneNames, typeMap, inPath, updateLog):
  if buglev >= 1: print 'doTree.entry: inPath: %s' % (inPath,)
  if os.path.isfile( inPath):
    if inPath.endswith('.cif'):
      if inPath in doneNames:
        if buglev >= 1: print 'main: already done: inPath: %s' % (inPath,)
      else:
        if buglev >= 1: print 'doTree: before  inPath: %s' % (inPath,)
        try:
          cifRdr = CifReader( buglev, typeMap, inPath)
          icsdMap = cifRdr.getIcsdMap()
        except CifException, exc:
          traceback.print_exc( None, sys.stdout)
          print "main: caught: %s" % (exc,)
        if buglev >= 1: print 'doTree: after   inPath: %s' % (inPath,)

        # Update the log
        fout = open( updateLog, 'a')
        print >> fout, inPath
        fout.close()
    else:
      if buglev >= 1: print 'doTree: not a cif: %s' % (inPath,)
  elif os.path.isdir( inPath):
    fnames = os.listdir( inPath)
    fnames.sort()
    for fn in fnames:
      subPath = '%s/%s' % (inPath, fn,)
      doTree( buglev, doneNames, typeMap, subPath, updateLog)   # recursion



#====================================================================


class CifReader:

  # Map: fieldName -> 'int' or 'float'

  typeMap = {
    '_database_code_ICSD'               : 'int',       # xxx stg?
    ##'_citation_journal_volume'          : 'listInt',
    ##'_citation_page_first'              : 'listInt',  # may be "174106-1"
    ##'_citation_page_last'               : 'listInt',
    ##'_citation_year'                    : 'listInt',

    '_cell_length_a'                    : 'float',
    '_cell_length_b'                    : 'float',
    '_cell_length_c'                    : 'float',
    '_cell_angle_alpha'                 : 'float',
    '_cell_angle_beta'                  : 'float',
    '_cell_angle_gamma'                 : 'float',
    '_cell_volume'                      : 'float',
    '_cell_formula_units_Z'             : 'int',
    '_symmetry_Int_Tables_number'       : 'int',
    '_atom_type_oxidation_number'       : 'listFloat',
    '_atom_site_symmetry_multiplicity'  : 'listInt',
    '_atom_site_fract_x'                : 'listFloat',
    '_atom_site_fract_y'                : 'listFloat',
    '_atom_site_fract_z'                : 'listFloat',
    '_atom_site_occupancy'              : 'listFloat',
    '_atom_site_attached_hydrogens'     : 'listInt',
    '_atom_site_B_iso_or_equiv'         : 'listFloat',
    '_symmetry_equiv_pos_site_id'       : 'listInt',
  }


  def __init__(self, buglev, inFile):

    # Note: don't strip lines.
    # We need to see the trailing spaces.
    # See the kluge for icsd_159700.cif.
    self.buglev = buglev
    self.inFile = inFile
    fin = open( inFile)
    self.lines = fin.readlines()
    fin.close()

    self.nline = len( self.lines)
    self.iline = 0         # current line num
    if self.nline == 0: self.line = None
    else: self.line = self.lines[0]

    self.nline = len( self.lines)
    self.cifMapDone = False
    self.icsdMapDone = False
    self.vaspMapDone = False
    self.errorList = []       # list of tuples: [(errorCode, errorMsg, iline)]

  def throwerr( self, msg):
    fullMsg = 'Error: %s  iline: %d' % (msg, self.iline)
    if self.line != None:
      fullMsg += '  line: %s' % ( repr( self.line),)
    raise Exception( fullMsg)

  def throwcif( self, msg):
    fullMsg = 'Error: %s  iline: %d' % (msg, self.iline)
    if self.line != None:
      fullMsg += '  line: %s' % ( repr( self.line),)
    raise CifException( fullMsg)

  def noteError( self, errorCode, errorMsg):
    self.errorList.append( (errorCode, errorMsg, self.iline))


  # Return tuple: (numErrorMsgs, formattedErrorMsg)

  def formatErrorMsg( self):
    useLongFormat = True
    if useLongFormat:
      msgs = ['%s:%d' % (tup[0],tup[2]) for tup in self.errorList]
      msgJoin = ','.join( msgs)
    else:
      mmap = {}
      for tup in self.errorList:
        if not mmap.has_key( tup[0]): self.errorMap[msg] = 0
        self.errorMap[tup[0]] += 1
      keys = mmap.keys()
      keys.sort()
      # Make list of strings like: 'msg:numOccur'
      msgs = []
      for key in keys:
        msgs.append('%s:%d' % (key, mmap[key],))
      msgJoin = ','.join( msgs)
    return (len(self.errorList), msgJoin)

  def advanceLine( self, msg):
    self.iline += 1
    if self.iline > self.nline: self.throwerr('iline > nline')
    if self.iline == self.nline: self.line = None
    else: self.line = self.lines[ self.iline]
    if self.buglev >= 5 and msg != None: self.printLine( msg)

  def printLine( self, msg):
    print '%s: iline: %d  line: %s' % (msg, self.iline, repr(self.line),)

  def getCifMap( self):
    if not self.cifMapDone:
      self.mkCifMap()
      self.cifMapDone = True
    return self.cifMap

  def getIcsdMap( self):
    if not self.icsdMapDone:
      self.mkIcsdMap()
      self.icsdMapDone = True
    return self.icsdMap

  def getVaspMap( self):
    if not self.vaspMapDone:
      self.mkVaspMap()
      self.vaspMapDone = True
    return self.vaspMap


  #====================================================================


  def mkCifMap( self):
    if self.buglev >= 2: print 'mkCifMap: entry'

    # Scan down for '_data'
    while self.iline < self.nline and \
      (self.line == '' or self.line.startswith('#')):
      self.advanceLine('mkCifMap: hd')

    if self.iline >= self.nline:
      self.throwerr('file is empty')
    if self.buglev >= 5: self.printLine('mkCifMap.a')

    self.cifMap = {}    # name/value map

    if not self.line.startswith('data_'):
      self.throwcif('no data stmt at start')
    dsname = self.line[5:].strip()
    if len(dsname) == 0: self.throwcif('dsname len == 0')
    self.cifMap['dsname'] = dsname
    self.advanceLine('mkCifMap: got dsname')

    while self.iline < self.nline:
      if len(self.line.strip()) == 0 or self.line.startswith('#'):
        self.advanceLine('mkCifMap: skip comment')
      elif self.line.rstrip() == 'loop_':
        self.advanceLine('mkCifMap: start loop')
        self.readLoop()
      else: self.readStandard()

    (self.cifMap['numError'], self.cifMap['errorMsg']) \
      = self.formatErrorMsg()

    if self.buglev >= 2:
      print '\nmkCifMap: cifMap:'
      keys = self.cifMap.keys()
      keys.sort()
      for key in keys:
        print '  key: %s  value: %s  (%s)' \
          % (key, repr( self.cifMap[key]), type( self.cifMap[key]).__name__,)


    if self.buglev >= 2: print 'mkCifMap: exit'

  #====================================================================


  # Read a standard (not loop) cif item.

  def readStandard( self):
    if self.buglev >= 5: self.printLine('readStandard.entry')
    if not self.line.startswith('_'): self.throwcif('invalid name')

    ix = self.line.find(' ')

    # _name value  or  _name 'value'  or  _name 'value'
    if ix >= 0:
      name = self.line[:ix].strip()
      stg = self.line[ix:].strip()
      if stg.startswith('"'):
        if not stg.endswith('"'): self.throwcif('no ending quote')
        stg = stg[1:-1]
      elif stg.startswith('\''):
        if not stg.endswith('\''): self.throwcif('no ending quote')
        stg = stg[1:-1]
      value = self.stripUncert( stg)
      self.advanceLine('readStandard: got single value')

    # _name\nvalue  or  _name\n;value ... ;
    else:
      name = self.line.strip()
      self.advanceLine('readStandard: start multiline')

      if self.iline >= self.nline: self.throwcif('value past end of file')
      # Value is delimited by ';'
      if self.line.startswith(';'):
        value = self.line[1:].strip()  # get remainder of initial ';' line
        self.advanceLine('readStandard: pass initial semicolon')
        while True:
          if self.iline >= self.nline: self.throwcif('value past end of file')
          if self.line.startswith(';'):
            if self.line.rstrip('\r\n') != ';':
              self.throwcif('invalid ending semicolon')
            self.advanceLine('got last semicolon')
            break
          if len(value) > 0: value += ' '
          value += self.line.strip()

          self.advanceLine('readStandard: multiline')

      # No ';' -- entire value is on next line
      else:
        value = self.stripUncert( self.line)

    cleanVal = self.cleanString( value)
    self.cifMap[name] = self.convertType( name, cleanVal)
    if self.buglev >= 5: self.printLine('readStandard.exit')



  #====================================================================


  # Read a loop cif item.

  def readLoop( self):
    if self.buglev >= 5: self.printLine('readLoop.entry')

    # Read the keys
    keys = []
    while True:
      if self.iline >= self.nline: self.throwcif('loop past end of file')
      if self.buglev >= 5: self.printLine('readLoop.a')
      if not self.line.startswith('_'): break
      keys.append( self.line.strip())
      self.advanceLine('readLoop: got key')
    nkey = len( keys)
    if self.buglev >= 5: print '  keys: %s' % (keys,)

    # Make list of empty sublists.  One sublist per key.
    valmat = []
    for ii in range(nkey):
      valmat.append([])

    # Read values
    loopDone = False
    while not loopDone:
      if self.buglev >= 5: self.printLine('readLoop.b')
      values = []      # values for this one line
      while len(values) < nkey:
        # Test for end
        if   self.line == None \
          or self.line.startswith('loop_') \
          or self.line.startswith('_') \
          or self.line.startswith('#'):
          loopDone = True
          break

        if self.iline >= self.nline: self.throwcif('value past end of file')
        if self.line.startswith(';'):
          value = self.line[1:].strip()  # get remainder of initial ';' line
          self.advanceLine('readLoop: pass initial semicolon')
          while True:
            if self.iline >= self.nline:
              self.throwcif('value past end of file')
            if self.line.startswith(';'):
              if self.line.rstrip('\r\n') != ';':
                self.throwcif('invalid ending semicolon')
              self.advanceLine('got last semicolon')
              break
            if len(value) > 0: value += ' '
            value += self.line.strip()
            self.advanceLine('readLoop: multiline')

          vals = [value]
          if self.buglev >= 5: print '    got semicolon value: ', vals

        else:     # else no ';'
          if self.iline >= self.nline: self.throwcif('value past end of file')
          tmpline = self.line

          # Kluge.  Sometimes have two spaces '  ' instead of ' 0 '.
          # Example: icsd_418816.cif
          if keys[0] == '_atom_site_label':
            ix = tmpline.find('  ')
            if ix >= 0:
              tmpline = tmpline[:ix] + ' 0 ' + tmpline[(ix+2):]
              self.noteError('atomMiss0',
                'readLoop: missing 0 in atom_site loop')

          vals = shlex.split( tmpline)     # split, retaining quoted substrings

          # Kluge: some lines in the citation loop look like:
          #   2 'Phase Transition' 1992 38- 127 220 PHTRDP
          # The '38-' should be two tokens, '38' and '-',
          # for the _citation_journal_volume and _citation_journal_issue.
          # Or
          #   1970 131- 139 146 ZEKGAX
          # The '131-' should be two tokens, '131' and '-'.
          if keys[0] == '_citation_id':
            for ix in range( len( vals)):
              if re.match('^[0-9]+-$', vals[ix]):
                vals[ix] = vals[ix][:-1]
                vals.insert( ix+1, '-')
                self.noteError('citeNoSpace',
                  'readLoop: no space before - in citation loop')
          
          # Kluge: sometimes the _atom_site_attached_hydrogens
          # is '-' instead of '0'.
          # Example: icsd_163077.cif
          if len(keys) >= 9 and keys[8] == '_atom_site_attached_hydrogens' \
            and len(vals) >= 9 and vals[8] == '-':
            vals[8] = '0'
            self.noteError('hydroDash',
              'readLoop: - instead of 0 for hydrogens')
          
          self.advanceLine('readLoop: pass single line record')
          if self.buglev >= 5: print '    got line vals: ', vals

        values += vals
      # end while len(values) < nkey

      if self.buglev >= 5:
        print '  nkey: %d  nval: %s' % (nkey, len(values),)
        print '  values: %s' % (values,)

      # Kluge Fixups:

      # icsd_108051.cif
      #     primary 'Zeitschrift fuer Metallkunde' 1983 74 358 389 8 ZEMTAE
      #     The '8' is spurious
      if self.cifMap['_database_code_ICSD'] == '108051' \
        and keys[0] == '_citation_id' \
        and len(values) >= 7 \
        and values[0] == 'primary' \
        and values[6] == '8':
        del values[6]
        self.noteError('icsd108051', 'readLoop: fixup 108051')

      # icsd_108051.cif
      #   primary 'Journal of the American Chemical Society' 2004 126 38 11780 11780 11781 JACSAT
      #     The second '11780' is spurious
      if self.cifMap['_database_code_ICSD'] == '170563' \
        and keys[0] == '_citation_id' \
        and len(values) >= 7 \
        and values[0] == 'primary' \
        and values[6] == '11780':
        del values[6]
        self.noteError('icsd170563', 'readLoop: fixup 170563')

      # icsd_418816.cif
      if self.cifMap['_database_code_ICSD'] == '418816' \
        and keys[0] == '_atom_site_label' \
        and len(values) >= 7 \
        and values[0] == 'primary' \
        and values[6] == '11780':
        del values[6]
        self.noteError('icsd418816', 'readLoop: fixup 418816')

      if len(values) != 0:
        if len(values) != nkey:
          self.throwcif(
            'wrong num values. nkey: %d  nval: %d\n  keys: %s\n  values: %s' \
            % (nkey, len(values), keys, values,))
        # Get a value for each column
        for ii in range(nkey):
          uval = self.stripUncert( values[ii])
          cleanVal = self.cleanString( uval)
          valmat[ii].append( cleanVal)

    for ii in range(nkey):
      tvec = self.convertType( keys[ii], valmat[ii])
      self.cifMap[ keys[ii]] = tvec
      if self.buglev >= 5:
        print 'readLoop: loop final: key: %s  values: %s' \
          % (keys[ii], valmat[ii],)
    if self.buglev >= 5: self.printLine('readLoop.exit')


  #====================================================================

  # If value has a standard uncertainty, like '3.44(5)',
  # strip off the uncertainty and discard it.

  def stripUncert( self, stg):
    # Kluge: allow entries with no ending right paren.
    # For example in icsd_180377.cif,
    #   Si3 Si4+ 6 i 0.447403(11) -0.447403(11 0.30818(6) 0.8333 0 0.00624(8)
    mat = re.match('^([-0-9.]+)\([0-9]+\)?$', stg)
    if mat != None:
      if not stg.endswith(')'):    # If ending paren, complain.
        self.noteError('uncertParen',
          'stripUncert: no R paren in uncertainty: "%s"' % (stg,))
      stg = stg[ : len(mat.group(1))]      # strip off uncertainty

    return stg


#====================================================================

  # Strip out illegal chars.
  # Example: icsd_054779.cif
  #   _chemical_name_systematic     'Magnesium Silicide (5/6) - Beta<F0>'

  def cleanString( self, stg):
    hasError = False
    ii = 0
    while ii < len(stg):
      ix = ord(stg[ii])
      if not (ix >= 32 and ix <= 126):
        tup = euroCharMap.get( ix, None)
        if tup != None:
          # Should an ISO_8859-15 char be an error?  Probably not.
          #hasError = True
          stg = stg[:ii] + tup[1] + stg[(ii+1):]
        else:
          hasError = True
          stg = stg[:ii] + stg[(ii+1):]
      ii += 1

    if hasError: self.noteError('badChar',
      'cleanString: invalid char dec %d in stg: "%s"' % ( ix, stg,))

    return stg

#====================================================================


  # If name is in typeMap, convert value to the right type.
  # Else leave it as a string.

  def convertType( self, name, value):

    if self.typeMap.has_key( name):
      ftype = self.typeMap[ name]
      if ftype == 'int':
        try: tvalue = int( value)
        except Exception, exc:
          self.throwcif('mkCifMap: invalid int for field "%s". value: "%s"' \
            % (name, value,))
      elif ftype == 'float':
        try: tvalue = float( value)
        except Exception, exc:
          self.throwcif('mkCifMap: invalid float for field "%s". value: "%s"' \
            % (name, value,))
      elif ftype == 'listInt':
        if not isinstance( value, list): throwerr('value not a list')
        nval = len( value)
        tvalue = nval * [None]
        for ii in range(nval):
          try: tvalue[ii] = int( value[ii])
          except Exception, exc:
            self.throwcif('mkCifMap: invalid int for field "%s". value: "%s"' \
              % (name, value[ii],))
      elif ftype == 'listFloat':
        if not isinstance( value, list): throwerr('value not a list')
        nval = len( value)
        tvalue = nval * [None]
        for ii in range(nval):
          try: tvalue[ii] = float( value[ii])
          except Exception, exc:
            self.throwcif(
              'mkCifMap: invalid float for field "%s". value: "%s"' \
              % (name, value[ii],))
      else: throwerr('unknown fieldType: "%s"' % (ftype,))
    else: tvalue = value

    return tvalue

#====================================================================


  # Sets self.icsdMap

  def mkIcsdMap(self):

    timea = datetime.datetime.now()
    if self.buglev >= 2: print 'mkIcsdMap: entry'

    self.getCifMap()          # insure self.cifMap is built

    # Check for required fields in cifMap
    requiredFields = [
      '_database_code_ICSD',
      '_chemical_name_systematic',
      '_chemical_formula_structural',
      '_chemical_formula_sum',
      '_cell_length_a',
      '_cell_length_b',
      '_cell_length_c',
      '_cell_angle_alpha',
      '_cell_angle_beta',
      '_cell_angle_gamma',
      '_cell_volume',
      '_cell_formula_units_Z',
      '_symmetry_space_group_name_H-M',
      '_symmetry_Int_Tables_number',
      '_atom_type_symbol',
      '_atom_type_oxidation_number',
      '_atom_site_label',
      '_atom_site_type_symbol',
      '_atom_site_symmetry_multiplicity',
      '_atom_site_Wyckoff_symbol',
      '_atom_site_fract_x',
      '_atom_site_fract_y',
      '_atom_site_fract_z',
      '_atom_site_occupancy',
      '_atom_site_attached_hydrogens',
    ]
    missFields = []
    for fld in requiredFields:
      if self.cifMap.get( fld, None) == None: missFields.append( fld)
    if len(missFields) > 0:
      self.throwcif('icsdMap: missing required fields: %s' % (missFields,))

    # Optional fields
    optionalFields = [
      '_chemical_name_mineral',
    ]
    missFields = []
    for fld in optionalFields:
      if self.cifMap.get( fld, None) == None: missFields.append( fld)
    if len(missFields) > 0:
      ##print('icsdMap: missing optional fields: %s' % (missFields,))
      pass

    elements = [
      'H',   'He',  'Li',   'Be',  'B',    'C',   'N',    'O',    'F',   'Ne',
      'Na',  'Mg',  'Al',   'Si',  'P',    'S',   'Cl',   'Ar',   'K',   'Ca',
      'Sc',  'Ti',  'V',    'Cr',  'Mn',   'Fe',  'Co',   'Ni',   'Cu',  'Zn',
      'Ga',  'Ge',  'As',   'Se',  'Br',   'Kr',  'Rb',   'Sr',   'Y',   'Zr',
      'Nb',  'Mo',  'Tc',   'Ru',  'Rh',   'Pd',  'Ag',   'Cd',   'In',  'Sn',
      'Sb',  'Te',  'I',    'Xe',  'Cs',   'Ba',  'La',   'Ce',   'Pr',  'Nd',
      'Pm',  'Sm',  'Eu',   'Gd',  'Tb',   'Dy',  'Ho',   'Er',   'Tm',  'Yb',
      'Lu',  'Hf',  'Ta',   'W',   'Re',   'Os',  'Ir',   'Pt',   'Au',  'Hg',
      'Tl',  'Pb',  'Bi',   'Po',  'At',   'Rn',  'Fr',   'Ra',   'Ac',  'Th',
      'Pa',  'U',   'Np',   'Pu',  'Am',   'Cm',  'Bk',   'Cf',   'Es',  'Fm',
      'Md',  'No',  'Lr',   'Rf',  'Db',   'Sg',  'Bh',   'Hs',   'Mt',  'Ds',
      'Rg',  'Cn',  'Uut',  'Fl',  'Uup',  'Lv',  'Uus',  'Uuo',
      'D', # Deuterium
    ]

    ferromagneticElements = ['Co', 'Cr', 'Fe', 'Ni']

    icmap = self.cifMap.copy()

    # Using the 'chemical_formula_sum' like 'Mo Se2',
    # get formulaNames = ['Mo', 'Se'] and formulaNums = [1, 2].
    # The formulaNums may be floats.
    # Kluge: sometimes the formulas omit a space, like icsd_171803.cif
    # '... Nb0.04 O20 Si4.49Ta0.01 Ti0.59'
    # So instead of using toks = chemSum.split()
    # we must find the tokens by scanning.

    chemSum = icmap['_chemical_formula_sum']           # 'Mo Se2'
    formulaNames = []
    formulaNums = []
    stg = chemSum
    while True:
      stg = stg.strip()
      if len(stg) == 0: break
      ##mata = re.match(r'^([a-zA-Z]+) *([.0-9]+)', stg)   # 'Se2'  or Ca1.37
      mata = re.match(r'^([a-zA-Z]+)([.0-9]+)', stg)   # 'Se2'  or Ca1.37
      if mata != None:
        formulaNames.append( mata.group(1))
        formulaNums.append( float( mata.group(2)))
        stg = stg[ len(mata.group(0)): ]
        # If no space before the next group, complain.
        if len(stg) > 0 and not stg.startswith(' '):
          self.noteError('chemSumSpace',
            'mkIcsdMap: no space in chemSum: "%s"' % (chemSum,))
      else:
        matb = re.match(r'^([a-zA-Z]+)', stg)   # 'Mo'
        if mata != None:
          formulaNames.append( mata.group(1))
          formulaNums.append( 1.0)
          stg = stg[ len(matb.group(0)): ]
        else: throwcif('unknown chem form sum: "%s"' % (chemSum,), None, None)

    for nm in formulaNames:
      if nm not in elements:
        throwcif('unknown element "%s" in chem form sum: "%s"' \
          % (nm, chemSum,), None, None)

    icmap['formulaNames'] = formulaNames
    icmap['formulaNums'] = formulaNums

    # Make a new field _atom_site_slabel = stripped atom_site_label.
    # Strip arbitrary trailing integer from label 'Mo1' to get 'Mo'.
    # Labels are sequential: Mg1, Mg2, Mg3, ...
    labels  = icmap['_atom_site_label']
    nlabel = len(labels)
    slabels = nlabel * [None]
    for ii in range( nlabel):
      slabels[ii] = re.sub('\d+$', '', labels[ii])
    icmap['_atom_site_slabel'] = slabels

    # Make a new field _atom_site_oxidation_num,
    # which is the oxidation number for each atom in _atom_site_type_symbol.
    syms = icmap['_atom_site_type_symbol']
    nsite = len( syms)
    ox_nums = nsite * [None]
    for ii in range( nsite):
      tsym = syms[ii]
      jj = icmap['_atom_type_symbol'].index( tsym)
      ox_nums[ii] = icmap['_atom_type_oxidation_number'][jj]
    icmap['_atom_site_oxidation_num'] = ox_nums

    # Set numCellAtom = total num atoms in cell
    mults   = icmap['_atom_site_symmetry_multiplicity']
    icmap['numCellAtom'] = sum( mults)

    # Set numCellFerro = total num ferromagnetic atoms in cell
    numferro = 0
    for ii in range( nlabel):
      if slabels[ii] in ferromagneticElements: numferro += mults[ii]
    icmap['numCellFerro'] = numferro

    # Max delta of formulaNums from integers
    fnums = icmap['formulaNums']
    fnumDelta = 0
    for fnum in fnums:
      dif = fnum % 1
      fnumDelta = max( fnumDelta, min( dif, 1 - dif))
    icmap['formulaDelta'] = fnumDelta

    # Max delta of occupancies from integers
    occs = icmap['_atom_site_occupancy']
    occuDelta = 0
    for occ in occs:
      dif = occ % 1
      occuDelta = max( occuDelta, min( dif, 1 - dif))
    icmap['occuDelta'] = occuDelta

    self.icsdMap = icmap

    (self.icsdMap['numError'], self.icsdMap['errorMsg']) \
      = self.formatErrorMsg()

    if self.buglev >= 2:
      print '\nmkIcsdMap: icsdMap:'
      keys = self.icsdMap.keys()
      keys.sort()
      for key in keys:
        print '  key: %s  value: %s  (%s)' \
          % (key, repr( self.icsdMap[key]), type( self.icsdMap[key]).__name__,)

    timeb = datetime.datetime.now()
    if self.buglev >= 1:
      print 'mkIcsdMap: icsd: %7d  num: %4d  chemSum: %s' \
        % (self.icsdMap['_database_code_ICSD'],
        self.icsdMap['numCellAtom'],
        self.icsdMap['_chemical_formula_sum'],)
    if self.buglev >= 2: print 'mkIcsdMap: %20s time: %10.5f' \
      % ('all', (timeb - timea).total_seconds(),)
    timea = timeb
    if self.buglev >= 2: print 'mkIcsdMap: exit'



  #====================================================================


  # Sets self.vaspMap

  def mkVaspMap( self):

    timea = datetime.datetime.now()
    if self.buglev >= 2: print 'mkVaspMap: entry'

    self.getIcsdMap()          # insure self.icsdMap is built

    # Cell
    # Adapted from crystal/read.py:icsd_cif
    lena  = self.icsdMap['_cell_length_a']
    lenb  = self.icsdMap['_cell_length_b']
    lenc  = self.icsdMap['_cell_length_c']
    alpha = self.icsdMap['_cell_angle_alpha']
    beta  = self.icsdMap['_cell_angle_beta']
    gamma = self.icsdMap['_cell_angle_gamma']

    a1 = lena * np.array([1.,0.,0.])
    a2 = lenb * np.array([np.cos(gamma*np.pi/180.),np.sin(gamma*np.pi/180.),0.])
    c1 = lenc * np.cos( beta * np.pi/180.)
    c2 = lenc / np.sin(gamma*np.pi/180.) * (-np.cos(beta*np.pi/180.) \
      * np.cos(gamma*np.pi/180.) + np.cos(alpha*np.pi/180.))
    a3 = np.array([c1, c2, np.sqrt(lenc**2-(c1**2+c2**2))])
    cell = np.array([a1,a2,a3])    # a1, a2, a3 are the rows.

    if self.buglev >= 2:
      print 'mkVaspMap: a1: %s' % (a1,)
      print 'mkVaspMap: a2: %s' % (a2,)
      print 'mkVaspMap: a3: %s' % (a3,)
      print 'mkVaspMap: cell:\n%s' % (cell,)

    # Symmetry ops: _symmetry_equiv_pos_as_xyz
    # Set transMats = list of 3x4 transformation matrices
    symStgs = self.icsdMap['_symmetry_equiv_pos_as_xyz']
    if self.buglev >= 5: print '\nmkVaspMap: symStgs: %s\n' % ( symStgs,)
    # ['x, x-y, -z+1/2', '-x+y, y, -z+1/2', ...]
    transMats = []
    for stg in symStgs:
      transMat = []
      specs = stg.split()
      if len(specs) != 3: throwcif('bad symStg: %s' % (stg,), None, None)
      for ii in range(len(specs)):
        specs[ii] = specs[ii].rstrip(',')    # get rid of trailing comma
        values = symParser( specs[ii])
        if self.buglev >= 5: print 'mkVaspMap: ii: %d  spec: %s  values: %s' \
          % (ii, specs[ii], values,)
        transMat.append( values)
      transMat = np.array( transMat, dtype=float)
      if self.buglev >= 10: print '\nmkVaspMap: transMat:\n%s\n' % ( transMat,)
      transMats.append( transMat)
    timeb = datetime.datetime.now()
    if self.buglev >= 2: print 'mkVaspMap: %20s time: %10.5f' \
      % ('symmetry ops', (timeb - timea).total_seconds(),)
    timea = timeb

    # atom_site positions
    slabels = self.icsdMap['_atom_site_slabel']    # labels w/o numeric suffix
    types   = self.icsdMap['_atom_site_type_symbol']
    symbols = self.icsdMap['_atom_site_Wyckoff_symbol']
    fracxs  = self.icsdMap['_atom_site_fract_x']
    fracys  = self.icsdMap['_atom_site_fract_y']
    fraczs  = self.icsdMap['_atom_site_fract_z']
    occs    = self.icsdMap['_atom_site_occupancy']
    hyds    = self.icsdMap['_atom_site_attached_hydrogens']

    nlabel = len(slabels)

    # Set wyckoffs = [
    #   ['Mo', ['0.3333', '0.6667', '0.25']],
    #   ['S', ['0.3333', '0.6667', '0.621']]
    # ]
    wyckoffs = []
    for ii in range( nlabel):
      wyc = [ slabels[ii], [ fracxs[ii], fracys[ii], fraczs[ii] ]]
      wyckoffs.append( wyc)
      if self.buglev >= 2: print 'mkVaspMap: wyc: %s' % (wyc,)

    # Get list of unique symbols: ['Mo', 'S']
    syms = [ww[0] for ww in wyckoffs]
    uniqueSyms = list( set( syms))               # unique values
    if self.buglev >= 2: print 'mkVaspMap: uniqueSyms: %s' % (uniqueSyms,)
    timeb = datetime.datetime.now()
    if self.buglev >= 2: print 'mkVaspMap: %20s time: %10.5f' \
      % ('unique syms', (timeb - timea).total_seconds(),)
    timea = timeb

    dtma = 0
    dtmb = 0
    dtmc = 0
    dtmd = 0


    # Set posVecs = list of sublists, one sublist per uniqueSym.
    # Each sublist is a list of unique position vectors,
    # generated by:
    #   posvec = np.dot( transMat, wycPos)
    # where
    #   transmat is in transmats, generated by the symmetry ops above
    #   wycPos comes from wyckoffs.

    if self.buglev >= 2: print '\nmkVaspMap: Get posVecs'
    posVecs = []                     # parallel array with uniqueSyms
    for ii in range(len(uniqueSyms)):
      posVecs.append( [])

    for wyc in wyckoffs:
      if self.buglev >= 2: print '  wyc: ', wyc
      wycSym = wyc[0]
      wycPos = wyc[1] + [1.0]
      for transMat in transMats:
        tma = datetime.datetime.now()
        if self.buglev >= 10: print '    transMat:\n%s' % ( transMat,)
        if self.buglev >= 10: print '    wycPos: %s' % (wycPos,)

        posVec = np.dot( transMat, wycPos)
        if self.buglev >= 10: print '    raw posVec: %s' % (posVec,)

        # Insure posVec elements are in the unit cube
        for ii in range(len(posVec)):
          if posVec[ii] < 0: posVec[ii] += 1
          if posVec[ii] >= 1: posVec[ii] -= 1
        if self.buglev >= 10: print '    final posVec: %s' % (posVec,)
        tmb = datetime.datetime.now()
        dtma += (tmb - tma).total_seconds()
        tma = tmb

        # If posVec is new, append posVec to positions for this symbol.
        ix = uniqueSyms.index( wycSym)

        tmb = datetime.datetime.now()
        dtmb += (tmb - tma).total_seconds()
        tma = tmb

        if posVecs[ix] == []:
          if self.buglev >= 5: print '    append posVec a: %s' % (posVec,)
          posVecs[ix].append( posVec)
        else:
          # If posVec is close to an ele already in posVecs, ignore posVec.
          # The following is too slow: about 20 seconds for 2000 ions.
          #   norms = [np.linalg.norm( oldVec - posVec)
          #     for oldVec in posVecs[ix]]
          #   if min( norms) < 0.01: ...
          #
          # So find a faster way.
          # The way used below takes about 15 seconds for 2000 ions.
          #
          # Perhaps in the future we could use a way that
          # keeps each vector posVecs[ix] in sorted order.
          # Then we can use a single binary search for dual purpose:
          # to see if a duplicate exists, and find the insertion point.

          minNorm = np.Infinity
          for pv in posVecs[ix]:
            norm = 0
            for ii in range( len( posVec)):
              delta = posVec[ii] - pv[ii]
              norm += delta * delta
            norm = np.sqrt( norm)
            if norm < minNorm: minNorm = norm


          tmb = datetime.datetime.now()
          dtmc += (tmb - tma).total_seconds()
          tma = tmb

          if minNorm < 0.01:
            if self.buglev >= 5: print '    duplicate posVec: %s' % (posVec,)
          else:
            if self.buglev >= 5: print '    append posVec b: %s' % (posVec,)
            posVecs[ix].append( posVec)

          tmb = datetime.datetime.now()
          dtmd += (tmb - tma).total_seconds()
          tma = tmb

    # Write posvecs to file 'tempplot'.
    writePlot = False
    if writePlot:
      fout = open( 'tempplot', 'w')
      for ii in range(len(uniqueSyms)):
        print 'mkVaspMap: posvecs for %s:' % (uniqueSyms[ii],)
        for pvec in posVecs[ii]:
          print '  %s' % (pvec,)
          print >> fout, '%s %s' % (ii, pvec,)
      fout.close()

    timeb = datetime.datetime.now()
    if self.buglev >= 2:
      print 'mkVaspMap: %20s time: %10.5f' \
        % ('posVecs', (timeb - timea).total_seconds(),)
      print 'mkVaspMap: dtma time: %10.5f' % (dtma,)
      print 'mkVaspMap: dtmb time: %10.5f' % (dtmb,)
      print 'mkVaspMap: dtmc time: %10.5f' % (dtmc,)
      print 'mkVaspMap: dtmd time: %10.5f' % (dtmd,)

    cellTrans = np.transpose( cell)

    if self.buglev >= 5:
      print 'mkVaspMap: cell:\n%s' % (cell,)
      print 'mkVaspMap: cellTrans:\n%s' % (cellTrans,)
      print '\nmkVaspMap: posVecs:'
      for ii in range(len(uniqueSyms)):
        print '  posVecs for ii: %d  sym: %s' % (ii, uniqueSyms[ii],)
        for posVec in posVecs[ii]:
          print '    posVec: %s' % (posVec,)

    atomVecs = []
    for ii in range(len(uniqueSyms)):
      atomVecs.append([])
      for posVec in posVecs[ii]:
        atomVec = np.dot( cellTrans, posVec)
        atomVecs[-1].append( atomVec)

    if self.buglev >= 5:
      print '\nmkVaspMap: atomVecs:'
      for ii in range(len(uniqueSyms)):
        print '  atomVecs for ii: %d  sym: %s' % (ii, uniqueSyms[ii],)
        for atomVec in atomVecs[ii]:
          print '    atomVec: %s' % (atomVec,)

    self.vaspMap = {}
    self.vaspMap['cellBasis'] = cell
    self.vaspMap['uniqueSyms'] = uniqueSyms
    self.vaspMap['posVecs'] = posVecs
    self.vaspMap['posScaleFactor'] = 1.0

    timeb = datetime.datetime.now()
    if self.buglev >= 2: print 'mkVaspMap: %20s time: %10.5f' \
      % ('finish', (timeb - timea).total_seconds(),)
    timea = timeb

    if self.buglev >= 2:
      print '\nmkVaspMap: self.vaspMap:'
      keys = self.vaspMap.keys()
      keys.sort()
      for key in keys:
        print '  key: %s  value: %s  (%s)' \
          % (key, repr( self.vaspMap[key]),
            type( self.vaspMap[key]).__name__,)

    if self.buglev >= 2: print 'mkVaspMap: exit'


#====================================================================

def writePoscar(
  sysName,
  vaspMap,
  outPoscar):            # output file name

  fout = open( outPoscar, 'w')

  # Name
  print >> fout, sysName

  # Universal scaling factor == lattice constant
  print >> fout, '%g' % (vaspMap['posScaleFactor'],)

  # Lattice vectors == basis vectors (rows) of the unit cell
  basis = vaspMap['cellBasis']
  for ii in range(3):
    for jj in range(3):
      print >> fout, ' %14.7g' % ( basis[ii,jj],),
    print >> fout, ''      # newline

  usyms = vaspMap['uniqueSyms']      # parallel array
  posVecs = vaspMap['posVecs']       # parallel array
  if len( usyms) != len( posVecs):
    throwerr('writePoscar: len(usyms) != len( posVecs)', None, None)

  # Num atoms of each species
  for ii in range( len( posVecs)):
    print >> fout, '%d' % (len( posVecs[ii]),),
  print >> fout, ''


  # Cartesian vs direct coords
  print >> fout, 'direct'

  # Atom positions
  for ii in range(len(usyms)):
    for posVec in posVecs[ii]:
      msg = ''
      for jj in range( len( posVec)):
        msg += ' %14.7g' % ( posVec[jj],)
      print >> fout, msg

  fout.close()


#====================================================================


def writePotcar(
  vaspMap,
  inPotenDir,
  outPotcar):           # output file name

  usyms = vaspMap['uniqueSyms']      # parallel array
  fout = open( outPotcar, 'w')
  for usym in usyms:
    fin = open( inPotenDir + '/' + usym + '/POTCAR')
    content = fin.read()
    fin.close()
    fout.write( content)
  fout.close()
    

  fout.close()



#====================================================================



# Lexer: x+0.22*y ==> 'x', '+', 0.22, '*', 'y'
def symLexer( stg):
  res = []
  while True:
    # Skip white space
    while len(stg) > 0 and stg[0] == ' ':
      stg.remove(0)

    if len(stg) == 0: break
    numstg = ''
    while len(stg) > 0 and (stg[0] >= '0' and stg[0] <= '9' or stg[0] == '.'):
      numstg += stg[0]
      stg = stg[1:]
    if len(numstg) > 0: res.append( float(numstg))
    else:
      res.append( stg[0])
      stg = stg[1:]
  return res


#====================================================================


# Parser: x+0.22*y ==> array with 4 elements: xCoeff, yCoeff, zCoeff, aconst
def symParser( stg):
  toks = symLexer( stg)
  names = 'xyz'

  # Simple syntax checks.
  # Essentially make sure operators are surrounded by operands,
  # and operands are surrounded by operators.
  tok = toks[0]                        # first token
  if not (tok in names or tok == '-' or type(tok == float)):
    throwcif('unknown syntax: "%s"' % (stg,), None, None)
  tok = toks[-1]                       # last token
  if not (type(tok) == float or tok in names):
    throwcif('unknown syntax: "%s"' % (stg,), None, None)
  for ii in range( 1, len(toks) - 1):  # all other tokens
    if type(toks[ii]) == float:
      if (toks[ii-1] not in '-+/') or (toks[ii+1] not in '-+/'):
        throwcif('unknown syntax: "%s"' % (stg,), None, None)
    elif toks[ii] in names:
      if (toks[ii-1] not in '-+') or (toks[ii+1] not in '-+'):
        throwcif('unknown syntax: "%s"' % (stg,), None, None)
    elif toks[ii] == '/':
      if type(toks[ii-1]) != float or type(toks[ii+1]) != float:
        throwcif('unknown syntax: "%s"' % (stg,), None, None)
    elif toks[ii] in '-+':
      if type(toks[ii-1]) != float and toks[ii-1] not in names:
        throwcif('unknown syntax: "%s"' % (stg,), None, None)
      if type(toks[ii+1]) != float and toks[ii+1] not in names:
        throwcif('unknown syntax: "%s"' % (stg,), None, None)
    else: throwcif('unknown syntax: "%s"' % (stg,), None, None)

  # Replace [vala, /, valb]   with   [vala/valb]
  ii = 0
  while ii < len(toks):
    if toks[ii] == '/':
      toks[ii-1] = toks[ii-1] / toks[ii+1]
      del toks[ii:(ii+2)]
    ii += 1

  # Scan for our symbols
  values = 4 * [None]
  for ii in range(len(toks)):
    tok = toks[ii]
    if type(tok) == float:
      if values[3] != None: throwcif('unknown syntax: "%s"' \
        % (stg,), None, None)
      values[3] = tok
    elif tok in names:
      inm = names.find( tok)
      if values[inm] != None: throwcif('unknown syntax: "%s"' \
        % (stg,), None, None)
      if ii == 0: values[inm] = 1.0      # leading 'x' or 'y' or 'z'
      elif toks[ii-1] == '-': values[inm] = -1.0
      elif toks[ii-1] == '+': values[inm] = 1.0
      else: throwcif('unknown syntax: "%s"' % (stg,), None, None)
    elif tok in '-+': pass
    else: throwcif('unknown syntax: "%s"' % (stg,), None, None)

  for ii in range(len(values)):
    if values[ii] == None: values[ii] = 0.
  return values


#====================================================================

# Convert from iso_8859-15 to ascii.  See man iso_8895-15.
# The default Postgresql database allows only ascii.
#
# Another approach would be to change the entire database
# by initializing it with
#   initdb -E LATIN9
# but that changes the entire database.
# We just need to handle 1 column of 1 table.

euroChars = [
  [ 160,  ' '       ,     'NO-BREAK SPACE'],
  [ 161,  '!'       ,     'INVERTED EXCLAMATION MARK'],
  [ 162,  ' c '     ,     'CENT SIGN'],
  [ 163,  ' pound ' ,     'POUND SIGN'],
  [ 164,  ' euro '  ,     'EURO SIGN'],
  [ 165,  ' yen '   ,     'YEN SIGN'],
  [ 166,  'S'       ,     'LATIN CAPITAL LETTER S WITH CARON'],
  [ 167,  ' sect '  ,     'SECTION SIGN'],
  [ 168,  's'       ,     'LATIN SMALL LETTER S WITH CARON'],
  [ 169,  'c'       ,     'COPYRIGHT SIGN'],
  [ 170,  ' '       ,     'FEMININE ORDINAL INDICATOR'],
  [ 171,  ' '       ,     'LEFT-POINTING DOUBLE ANGLE QUOTATION MARK'],
  [ 172,  ' not '   ,     'NOT SIGN'],
  [ 173,  '-'       ,     'SOFT HYPHEN'],
  [ 174,  ' reg '   ,     'REGISTERED SIGN'],
  [ 175,  '-'       ,     'MACRON'],
  [ 176,  ' deg '   ,     'DEGREE SIGN'],
  [ 177,  ' +/- '   ,     'PLUS-MINUS SIGN'],
  [ 178,  ' 2 '     ,     'SUPERSCRIPT TWO'],
  [ 179,  ' 3 '     ,     'SUPERSCRIPT THREE'],
  [ 180,  'Z'       ,     'LATIN CAPITAL LETTER Z WITH CARON'],
  [ 181,  'u'       ,     'MICRO SIGN'],
  [ 182,  ' P '     ,     'PILCROW SIGN'],
  [ 183,  '.'       ,     'MIDDLE DOT'],
  [ 184,  'z'       ,     'LATIN SMALL LETTER Z WITH CARON'],
  [ 185,  '  1 '    ,     'SUPERSCRIPT ONE'],
  [ 186,  'o'       ,     'MASCULINE ORDINAL INDICATOR'],
  [ 187,  ' > '     ,     'RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK'],
  [ 188,  'OE'      ,     'LATIN CAPITAL LIGATURE OE'],
  [ 189,  'oe'      ,     'LATIN SMALL LIGATURE OE'],
  [ 190,  'Y'       ,     'LATIN CAPITAL LETTER Y WITH DIAERESIS'],
  [ 191,  '?'       ,     'INVERTED QUESTION MARK'],
  [ 192,  'A'       ,     'LATIN CAPITAL LETTER A WITH GRAVE'],
  [ 193,  'A'       ,     'LATIN CAPITAL LETTER A WITH ACUTE'],
  [ 194,  'A'       ,     'LATIN CAPITAL LETTER A WITH CIRCUMFLEX'],
  [ 195,  'A'       ,     'LATIN CAPITAL LETTER A WITH TILDE'],
  [ 196,  'A'       ,     'LATIN CAPITAL LETTER A WITH DIAERESIS'],
  [ 197,  'A'       ,     'LATIN CAPITAL LETTER A WITH RING ABOVE'],
  [ 198,  'AE'      ,     'LATIN CAPITAL LETTER AE'],
  [ 199,  'C'       ,     'LATIN CAPITAL LETTER C WITH CEDILLA'],
  [ 200,  'E'       ,     'LATIN CAPITAL LETTER E WITH GRAVE'],
  [ 201,  'E'       ,     'LATIN CAPITAL LETTER E WITH ACUTE'],
  [ 202,  'E'       ,     'LATIN CAPITAL LETTER E WITH CIRCUMFLEX'],
  [ 203,  'E'       ,     'LATIN CAPITAL LETTER E WITH DIAERESIS'],
  [ 204,  'I'       ,     'LATIN CAPITAL LETTER I WITH GRAVE'],
  [ 205,  'I'       ,     'LATIN CAPITAL LETTER I WITH ACUTE'],
  [ 206,  'I'       ,     'LATIN CAPITAL LETTER I WITH CIRCUMFLEX'],
  [ 207,  'I'       ,     'LATIN CAPITAL LETTER I WITH DIAERESIS'],
  [ 208,  'D'       ,     'LATIN CAPITAL LETTER ETH'],
  [ 209,  'N'       ,     'LATIN CAPITAL LETTER N WITH TILDE'],
  [ 210,  'O'       ,     'LATIN CAPITAL LETTER O WITH GRAVE'],
  [ 211,  'O'       ,     'LATIN CAPITAL LETTER O WITH ACUTE'],
  [ 212,  'O'       ,     'LATIN CAPITAL LETTER O WITH CIRCUMFLEX'],
  [ 213,  'O'       ,     'LATIN CAPITAL LETTER O WITH TILDE'],
  [ 214,  'O'       ,     'LATIN CAPITAL LETTER O WITH DIAERESIS'],
  [ 215,  'x'       ,     'MULTIPLICATION SIGN'],
  [ 216,  'O'       ,     'LATIN CAPITAL LETTER O WITH STROKE'],
  [ 217,  'U'       ,     'LATIN CAPITAL LETTER U WITH GRAVE'],
  [ 218,  'U'       ,     'LATIN CAPITAL LETTER U WITH ACUTE'],
  [ 219,  'U'       ,     'LATIN CAPITAL LETTER U WITH CIRCUMFLEX'],
  [ 220,  'U'       ,     'LATIN CAPITAL LETTER U WITH DIAERESIS'],
  [ 221,  'Y'       ,     'LATIN CAPITAL LETTER Y WITH ACUTE'],
  [ 222,  ' '       ,     'LATIN CAPITAL LETTER THORN'],
  [ 223,  's'       ,     'LATIN SMALL LETTER SHARP S'],
  [ 224,  'a'       ,     'LATIN SMALL LETTER A WITH GRAVE'],
  [ 225,  'a'       ,     'LATIN SMALL LETTER A WITH ACUTE'],
  [ 226,  'a'       ,     'LATIN SMALL LETTER A WITH CIRCUMFLEX'],
  [ 227,  'a'       ,     'LATIN SMALL LETTER A WITH TILDE'],
  [ 228,  'a'       ,     'LATIN SMALL LETTER A WITH DIAERESIS'],
  [ 229,  'a'       ,     'LATIN SMALL LETTER A WITH RING ABOVE'],
  [ 230,  'ae'      ,     'LATIN SMALL LETTER AE'],
  [ 231,  'c'       ,     'LATIN SMALL LETTER C WITH CEDILLA'],
  [ 232,  'e'       ,     'LATIN SMALL LETTER E WITH GRAVE'],
  [ 233,  'e'       ,     'LATIN SMALL LETTER E WITH ACUTE'],
  [ 234,  'e'       ,     'LATIN SMALL LETTER E WITH CIRCUMFLEX'],
  [ 235,  'e'       ,     'LATIN SMALL LETTER E WITH DIAERESIS'],
  [ 236,  'i'       ,     'LATIN SMALL LETTER I WITH GRAVE'],
  [ 237,  'i'       ,     'LATIN SMALL LETTER I WITH ACUTE'],
  [ 238,  'i'       ,     'LATIN SMALL LETTER I WITH CIRCUMFLEX'],
  [ 239,  'i'       ,     'LATIN SMALL LETTER I WITH DIAERESIS'],
  [ 240,  'o'       ,     'LATIN SMALL LETTER ETH'],
  [ 241,  'n'       ,     'LATIN SMALL LETTER N WITH TILDE'],
  [ 242,  'o'       ,     'LATIN SMALL LETTER O WITH GRAVE'],
  [ 243,  'o'       ,     'LATIN SMALL LETTER O WITH ACUTE'],
  [ 244,  'o'       ,     'LATIN SMALL LETTER O WITH CIRCUMFLEX'],
  [ 245,  'o'       ,     'LATIN SMALL LETTER O WITH TILDE'],
  [ 246,  'o'       ,     'LATIN SMALL LETTER O WITH DIAERESIS'],
  [ 247,  '/'       ,     'DIVISION SIGN'],
  [ 248,  'o'       ,     'LATIN SMALL LETTER O WITH STROKE'],
  [ 249,  'u'       ,     'LATIN SMALL LETTER U WITH GRAVE'],
  [ 250,  'u'       ,     'LATIN SMALL LETTER U WITH ACUTE'],
  [ 251,  'u'       ,     'LATIN SMALL LETTER U WITH CIRCUMFLEX'],
  [ 252,  'u'       ,     'LATIN SMALL LETTER U WITH DIAERESIS'],
  [ 253,  'y'       ,     'LATIN SMALL LETTER Y WITH ACUTE'],
  [ 254,  ' '       ,     'LATIN SMALL LETTER THORN'],
  [ 255,  'y'       ,     'LATIN SMALL LETTER Y WITH DIAERESIS'],
]

euroCharMap = {}
for tup in euroChars:
  euroCharMap[ tup[0]] = tup

#====================================================================


if __name__ == '__main__': main()



#====================================================================
#====================================================================
#====================================================================

## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD
#
## Parse a symmetry operation string like '-x+y-z+1/2'.
#
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
#class Operator:
#  def __init__(self, symbol, precedence):
#    checkType( str, symbol)
#    checkType( int, precedence)
#    self.symbol = symbol
#    self.precedence = precedence
#  def __str__(self):
#    res = 'symbol: %s  precedence: %d' % (self.symbol, self.precedence,)
#    return res
#
#
#
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
#def parseSymOp(
#  buglev,
#  stgParm,
#  valueMap):
#
#  checkType( int, buglev)
#  checkType( str, symStg)
#  checkType( dict, valueMap)
#
#  operators = []
#  operators.append( Operator('-', 10))
#  operators.append( Operator('+', 10))
#  operators.append( Operator('/', 20))
#
#  specNames = 'xyz'
#  nspec = len( specNames)
#  specValues = nspec * [-999.0]
#
#  opStk = []
#  valStk = []
#  stg = stgParm
#  for istg in range(len(symStg)):
#    (stg, token) = getToken( stg)
#    if token != ' ':
#      iop = -1
#      for ii in range(len(operators)):
#        if token == operators[ii].symbol:
#          iop = ii
#          break
#      if iop >= 0:          # if an operator
#        oper = operators[iop]
#        if len(opStk) == 0 or opStk[0].prec < oper.prec:
#          if istg == 0: valStk.insert( 0, 0.)    # convert prefix ops to binops
#          opStk.insert( 0, oper)
#        else:
#          execOp( opStk, valStk)
#
#      else:                # else not an operator
#        if specNames.haskey( token):
#          ix = specNames.find( token)
#          if istg == 0:                 # leading x, y, or z
#            specValues[ix] = 1.0
#          elif opStk[0] == opPlus:      # '... +x'
#            specValues[ix] = 1.0
#          elif opStk[0] == opMinus:     # '... -x'
#            specValues[ix] = -1.0
#          elif opStk[0] == opStar:      # '..a*b/c * x'
#            specValues[ix] = -1.0
#          else: throwerr('unknown syntax')
#          valStk.insert( 0, valueMap[token])
#
#        else:              # not in specNames
#          valStk.insert( 0, valueMap[token])
#
#
#
##====================================================================
## OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
#def execOp(
#  buglev,
#  opStk,
#  valStk):
#
#  checkType( int, buglev)
#  checkType( list, opStk)
#  checkType( list, valStk)
#
#  op = opStk[0]
#  if   op == '-': valStk[1] -= valStk[0]
#  elif op == '+': valStk[1] += valStk[0]
#  elif op == '*': valStk[1] *= valStk[0]
#  elif op == '/': valStk[1] /= valStk[0]
#  else: throwerr('unknown op:  "%s"' % (op,))
#  valStk.remove( 0)
#  opStk.remove( 0)


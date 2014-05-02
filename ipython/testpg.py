
""" IPython testpg function and completer. """
def testpg(self, cmdl): 
  """ test whether pg has any clue about ipython magic functions """

  print "you were called with command line:"
  print cmdl
  print "now what !?"

def completer(self, event): 
    pass

  

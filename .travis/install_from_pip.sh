#! /usr/bin/env bash
set -e
set -o

if [ "$TRAVIS_PULL_REQUEST" = "false" ] ; then
  pip install --user git+git://github.com/$TRAVIS_REPO_SLUG@$TRAVIS_COMMIT
  python -c "import pylada; pylada.test()"
fi

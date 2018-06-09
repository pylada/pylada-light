#! /usr/bin/env bash
set -e
set -o

cat > install_from_pip.sh << EOF
#!/bin/bash
source ~/.bashrc
set -e
set -o
if [ "$TRAVIS_PULL_REQUEST" = "false" ] ; then
  python3 -m pip install --user git+git://github.com/$TRAVIS_REPO_SLUG@$TRAVIS_COMMIT
  python -c "import pylada; pylada.test()"
fi
EOF
chmod u+x install_from_pip.sh

docker run -it --rm -v $(pwd):/project -w /project \
        --env "CC=$ccomp" --env "CXX=$cxxcomp" --cap-add SYS_PTRACE \
        ./install_from_pip.sh

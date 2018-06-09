usage="Sets helper scripts to ease building pylada on the docker containers:

Usage
-----

pylada restart
   restarts the docker vm
   Unfortunately, this is often necessary on Mac, because file changes are not passed on to the VM
   where docker is running.

pylada setup (gcc|llvm) [args]
   sets up in docker_(gcc|llvm)
   Extra arguments are passed on to cmake

pylada build (gcc|llvm) targetname [args]
   build a given target
   Extra arguments are passed to the build system (make or ninja, presumably)

pylada test (gcc|llvm) [ctest options]
   run some tests

pylada (gcc|llvm) [args]
   runs a command in docker

pylada help
   this string"


function restart_pylada {
    docker-machine restart pylada
    eval $(docker-machine env pylada)
}

function cmd_pylada {

    [ "$#" -lt "1" ] && return

    name=$1
    if [ "$1" = "gcc" ] ; then
        ccomp="gcc"
        cxxcomp="g++"
        directory="/project/build_gcc"
        shift
    elif [ "$1" = "llvm" ] ; then
        ccomp="clang"
        cxxcomp="clang++"
        directory="/project/build_llvm"
        shift
    else
        ccomp="gcc"
        cxxcomp="g++"
        directory="/project/build_gcc"
    fi

    cat > build_$name/docker_script.sh <<EOF
#!/bin/bash -l
        set -o
        set -e
        module load mpi
        $@
EOF
    chmod u+x build_$name/docker_script.sh
    docker run -it --rm -v $(pwd):/project -w $directory \
        --env "CC=$ccomp" --env "CXX=$cxxcomp" --cap-add SYS_PTRACE \
        pylada:dnf ./docker_script.sh
}

function build_pylada {
    if [ "$1" = "gcc" ] ; then
        compiler="gcc"
        shift
    elif [ "$1" = "llvm" ] ; then
        compiler="llvm"
        shift
    else
        compiler="gcc"
    fi

    if [ "$#" -gt 0 ] ; then
        target=$1
        shift
        cmd_pylada $compiler cmake --build /project/build_$compiler --target $target -- $@
    else
        cmd_pylada $compiler cmake --build /project/build_$compiler --target all
    fi
}

function test_pylada {
    if [ "$1" = "gcc" ] ; then
        compiler="gcc"
        shift
    elif [ "$1" = "llvm" ] ; then
        compiler="llvm"
        shift
    else
        compiler="gcc"
    fi

    cmd_pylada $compiler ctest . $@
}

function setup_pylada {
    if [ "$1" = "gcc" ] ; then
        compiler="gcc"
        shift
    elif [ "$1" = "llvm" ] ; then
        compiler="llvm"
        shift
    else
        compiler="gcc"
    fi

    directory="build_$compiler"
    echo "Setting up $compiler in $directory"
    [ -e "$directory" ] || mkdir $directory
    cmd_pylada $compiler cmake $@ ..
}

function pylada {
    case $1 in
        "restart")
            shift
            restart_pylada $@
            ;;
        "setup")
            shift
            setup_pylada $@
            ;;
        "build")
            shift
            build_pylada $@
            ;;
        "test")
            shift
            test_pylada $@
            ;;
        "cmd")
            shift
            cmd_pylada $@
            ;;
        "help")
            echo $usage
            ;;
        "-h")
            echo $usage
            ;;
        *)
            cmd_pylada $@
            ;;
    esac
}

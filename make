#!/bin/bash

Help()
{
   # Display Help
   echo "Compile FiberMat files."
   echo "Basic use: make --all"
   echo
   echo "Syntax: make [-h|-a|-b|-t|-d|-f]"
   echo "Options:"
   echo "    -h | --help                                Display help."
   echo "    -a | --all                                 Compile sources, run doctest and compile documentation."
   echo "    -b | --build                               Compile sources."
   echo "    -t | --test | --doctest                    Run pytest and doctest."
   echo "    -d | --doc                                 Compile documentation."
   echo "    -f | --file | --files  file [file, ...]    Run doctest on given files."
   echo
}

here="."

f=false
p=false
d=false
f="$here/src/fibermat/*.py"

if [ "$#" = 0 ]
then
    Help
    exit
fi

while [[ $# -gt 0 ]]
do
    case "$1" in
    -h|--help)
        Help  # Display help
        exit
        ;;
    -a|--all)
        b=true
        t=true
        d=true
        ;;
    -b|--build)
        b=true  # Compile sources
        ;;
    -t|--test|--doctest)
        t=true  # Run doctests
        ;;
    -d|--doc)
        d=true  # Compile documentation
        ;;
    -f|--file|--files)
        f=""
        while [ "$1" ]
        do
            shift
            f="$f $1"
        done
        if [ "$f" = " " ]
        then
            echo "Error: the following arguments are required: file"
        fi
        ;;
    esac
    shift
done

if [ "$b" = true ]
then
    pip install --upgrade $here
    python -m build
fi

if [ "$t" = true ]
then
    python -m doctest -v $f
fi

if [ "$d" = true ]
then
    make html
fi

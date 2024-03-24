#!/bin/bash

Help()
{
   # Display Help
   echo "Compile FiberMat files."
   echo "Basic use: make --all"
   echo
   echo "Syntax: make [-h|-t|-p|-d|-a|-f]"
   echo "Options:"
   echo "    -h | --help                                Display help."
   echo "    -t | --test | --doctest                    Run doctest."
   echo "    -p | --pip                                 Compile sources."
   echo "    -d | --doc                                 Build documentation."
   echo "    -a | --all                                 Do all."
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
    -t|--test|--doctest)
        t=true  # Run doctests
        ;;
    -p|--pip)
        p=true  # Compile sources
        ;;
    -d|--doc)
        d=true  # Build documentation
        ;;
    -a|--all)
        t=true
        p=true
        d=true
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

if [ "$t" = true ]
then
    python -m doctest -v $f
fi

if [ "$p" = true ]
then
    pip install --upgrade $here
fi

if [ "$d" = true ]
then
    make html
fi

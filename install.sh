#!/bin/bash

shopt -s expand_aliases
set -e

OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" = "Linux" ]; then
    alias gx1="g++ -O3 -c -fpic -fopenmp "
    alias gx2="g++ -O3 -shared -fopenmp "
elif [ "${OS}" = "Darwin" ]; then
    if [ "${ARCH}" = "arm64" ]; then
        compiler="/opt/homebrew/opt/llvm/bin/clang++"
    else
        compiler="/usr/local/opt/llvm/bin/clang++"
    fi
    alias gx1="${compiler} -O3 -c -fpic -fopenmp "
    alias gx2="${compiler} -O3 -shared -fopenmp "
else
    echo "Unsupported OS"
    exit -1
fi

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o

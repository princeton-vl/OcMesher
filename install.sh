#!/bin/bash

shopt -s expand_aliases
set -e

OS=$(uname -s)
ARCH=$(uname -m)

if [ -n "$CXX" ]; then
    compiler="$CXX"
else
    if [ "${OS}" = "Linux" ]; then
        compiler="g++"
    elif [ "${OS}" = "Darwin" ]; then
        if [ "${ARCH}" = "arm64" ]; then
            compiler="/opt/homebrew/opt/llvm/bin/clang++"
        else
            compiler="/usr/local/opt/llvm/bin/clang++"
        fi
    else
        echo "Unsupported OS"
        exit -1
    fi
fi

alias gx1="${compiler} \$CXXFLAGS -O3 -c -fpic -fopenmp "
alias gx2="${compiler} \$LDFLAGS -O3 -shared -fopenmp "

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o

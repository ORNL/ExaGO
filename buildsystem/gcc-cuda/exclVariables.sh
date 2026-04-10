#!/bin/bash

SRCDIR=${SRCDIR:-$PWD}

# Platform specific configuration
source $SRCDIR/buildsystem/gcc-cuda/excl/base.sh

# Spack modules
source $SRCDIR/buildsystem/spack/excl/modules/dependencies.sh 

#!/bin/bash

export MY_CLUSTER=excl
. buildsystem/spack/load_spack.sh && \
spack develop --no-clone --path=$(pwd) exago@develop && \
buildsystem/spack/configure_modules.sh 32

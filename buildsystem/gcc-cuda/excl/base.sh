#!/bin/bash

export MY_CLUSTER=excl

module reset
module load gcc/12.2.0
module load openmpi

export EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_CUDA_HOST_COMPILER=gcc"

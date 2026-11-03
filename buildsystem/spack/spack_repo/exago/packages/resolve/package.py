# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.packages.resolve.package import Resolve as BuiltinResolve

from spack.package import *


class Resolve(BuiltinResolve):
    """Builtin ReSolve package, compiling HIP sources with hipcc."""

    depends_on("hip", when="+rocm")

    def cmake_args(self):
        args = super().cmake_args()

        # TODO: move to the builtin package, which leaves the C++ compiler as g++.
        if self.spec.satisfies("+rocm"):
            args.append(self.define("CMAKE_CXX_COMPILER", self.spec["hip"].hipcc))

        return args

module use -a /lustre/orion/stf006/world-shared/nkouk/exago-08-2026/spack-install/modules/linux-sles15-zen3
## excluded or missing from upstream: cmake@=3.30.5~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release platform=linux os=sles15 target=x86_64
# compiler-wrapper@=1.1.0 build_system=generic platform=linux os=sles15 target=zen3
module load compiler-wrapper/1.1.0-none-none-hsr6o73
## excluded or missing from upstream: gcc@=14.2.0+binutils+bootstrap~graphite+libsanitizer~mold~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: glibc@=2.38 build_system=autotools platform=linux os=sles15 target=x86_64
# gcc-runtime@=14.2.0 build_system=generic platform=linux os=sles15 target=zen3
module load gcc-runtime/14.2.0-none-none-mthljcr
# blt@=0.7.2 build_system=generic platform=linux os=sles15 target=zen3
module load blt/0.7.2-gcc-14.2.0-v5iinju
# gmake@=4.4.1~guile build_system=generic platform=linux os=sles15 target=zen3
module load gmake/4.4.1-gcc-14.2.0-r4kiolf
## excluded or missing from upstream: hip@=6.3.1~asan~cuda+rocm build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: hsa-rocr-dev@=6.3.1~asan+image+shared build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: llvm-amdgpu@=6.3.1~link_llvm_dylib~llvm_dylib+rocm-device-libs build_system=cmake build_type=Release generator=ninja languages:='c,c++' platform=linux os=sles15 target=x86_64
# camp@=2025.12.0~cuda~ipo~omptarget~openmp+rocm~sycl~tests amdgpu_target:=gfx90a build_system=cmake build_type=Release commit=a8caefa9f4c811b1a114b4ed2c9b681d40f12325 generator=make platform=linux os=sles15 target=zen3
module load camp/2025.12.0-gcc-14.2.0-2mvr5d4
## excluded or missing from upstream: cray-mpich@=8.1.31~cuda~rocm+wrappers build_system=generic platform=linux os=sles15 target=x86_64
# fmt@=11.0.2~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make platform=linux os=sles15 target=zen3
module load fmt/11.0.2-gcc-14.2.0-rymouym
## excluded or missing from upstream: python@=3.11.7+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~static~tests~tkinter+uuid+zlib build_system=generic platform=linux os=sles15 target=x86_64
# re2c@=4.4 build_system=autotools platform=linux os=sles15 target=zen3
module load re2c/4.4-gcc-14.2.0-zu5yo7f
# ninja@=1.13.2+re2c build_system=generic platform=linux os=sles15 target=zen3
module load ninja/1.13.2-gcc-14.2.0-wlhvs6o
# python-venv@=1.0 build_system=generic platform=linux os=sles15 target=zen3
module load python-venv/1.0-none-none-pfnl5ia
# py-pip@=26.1.2 build_system=generic platform=linux os=sles15 target=zen3
module load py-pip/26.1.2-none-none-y6j2d6n
# py-setuptools@=82.0.1 build_system=generic platform=linux os=sles15 target=zen3
module load py-setuptools/82.0.1-none-none-qk6gabk
# py-wheel@=0.45.1 build_system=generic platform=linux os=sles15 target=zen3
module load py-wheel/0.45.1-none-none-e5kilm7
# meson@=1.11.1 build_system=python_pip patches:=0f0b1bd platform=linux os=sles15 target=zen3
module load meson/1.11.1-none-none-e6yrimg
# metis@=5.1.0~gdb~int64~ipo~no_warning~real64+shared build_system=cmake build_type=Release generator=make patches:=4991da9,93a7903,b1225da platform=linux os=sles15 target=zen3
module load metis/5.1.0-gcc-14.2.0-gx4ixil
# berkeley-db@=18.1.40+cxx~docs+stl build_system=autotools patches:=26090f4,b231fcc platform=linux os=sles15 target=zen3
module load berkeley-db/18.1.40-gcc-14.2.0-hmndfu6
# libiconv@=1.18 build_system=autotools libs:=shared,static platform=linux os=sles15 target=zen3
module load libiconv/1.18-gcc-14.2.0-sidqw3h
# diffutils@=3.12 build_system=autotools platform=linux os=sles15 target=zen3
module load diffutils/3.12-gcc-14.2.0-v4ysmvr
# bzip2@=1.0.8~debug~pic+shared build_system=generic platform=linux os=sles15 target=zen3
module load bzip2/1.0.8-gcc-14.2.0-i2jpxnn
# pkgconf@=2.5.1 build_system=autotools platform=linux os=sles15 target=zen3
module load pkgconf/2.5.1-gcc-14.2.0-qs7ovds
# ncurses@=6.6~symlinks+termlib abi=none build_system=autotools patches:=7a351bc platform=linux os=sles15 target=zen3
module load ncurses/6.6-gcc-14.2.0-262vg53
# readline@=8.3 build_system=autotools patches:=21f0a03,72dee13,e273643 platform=linux os=sles15 target=zen3
module load readline/8.3-gcc-14.2.0-ubqvjhm
# gdbm@=1.26 build_system=autotools platform=linux os=sles15 target=zen3
module load gdbm/1.26-gcc-14.2.0-67rbhbv
# less@=692 build_system=autotools platform=linux os=sles15 target=zen3
module load less/692-gcc-14.2.0-4pc2lx7
# zlib-ng@=2.3.3+compat+new_strategies+opt+pic+shared build_system=autotools platform=linux os=sles15 target=zen3
module load zlib-ng/2.3.3-gcc-14.2.0-is5hyhl
# perl@=5.42.0+cpanm+opcode+open+shared+threads build_system=generic platform=linux os=sles15 target=zen3
module load perl/5.42.0-gcc-14.2.0-x3k7ozd
# openblas@=0.3.20~bignuma~consistent_fpcsr+dynamic_dispatch~ilp64+locking+pic+shared~static build_system=makefile patches:=9f12903 symbol_suffix=none threads=none platform=linux os=sles15 target=zen3
module load openblas/0.3.20-gcc-14.2.0-rbgtzyf
# coinhsl@=2024.05.15+metis~strip build_system=meson buildtype=release default_library:=shared platform=linux os=sles15 target=zen3
module load coinhsl/2024.05.15-gcc-14.2.0-fyo6bb6
## excluded or missing from upstream: hipblas@=6.3.1~asan~cuda+rocm amdgpu_target:=auto build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: hiprand@=6.3.1~asan~cuda+rocm amdgpu_target:=auto build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: hipsparse@=6.3.1~asan~cuda+rocm amdgpu_target:=auto build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
## excluded or missing from upstream: rocm-core@=6.3.1~asan build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
# magma@=2.8.0~cuda+fortran~ipo+rocm+shared amdgpu_target:=gfx90a build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=zen3
module load magma/2.8.0-gcc-14.2.0-pi6nfpp
## excluded or missing from upstream: rocprim@=6.3.1~asan amdgpu_target:=auto build_system=cmake build_type=Release generator=make platform=linux os=sles15 target=x86_64
# raja@=2025.12.2~caliper~cuda~desul~examples~exercises~gpu-profiling~ipo~lowopttest~omptarget~omptask~openmp~plugins+rocm~run-all-tests~shared~sycl~tests+vectorization amdgpu_target:=gfx90a build_system=cmake build_type=Release commit=eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274 cxxstd=20 generator=make platform=linux os=sles15 target=zen3
module load raja/2025.12.2-gcc-14.2.0-u56663i
# libsigsegv@=2.15 build_system=autotools platform=linux os=sles15 target=zen3
module load libsigsegv/2.15-gcc-14.2.0-gt2mrdc
# m4@=1.4.21+sigsegv build_system=autotools platform=linux os=sles15 target=zen3
module load m4/1.4.21-gcc-14.2.0-fjjki5v
# autoconf@=2.72 build_system=autotools platform=linux os=sles15 target=zen3
module load autoconf/2.72-none-none-lnrszbp
# automake@=1.18.1 build_system=autotools platform=linux os=sles15 target=zen3
module load automake/1.18.1-gcc-14.2.0-aw4o7fg
# xz@=5.8.3~pic build_system=autotools libs:=shared,static platform=linux os=sles15 target=zen3
module load xz/5.8.3-gcc-14.2.0-5vrw34f
# zstd@=1.5.7+programs build_system=makefile compression:=none libs:=shared,static platform=linux os=sles15 target=zen3
module load zstd/1.5.7-gcc-14.2.0-uytpcd3
# file@=5.46+static build_system=autotools platform=linux os=sles15 target=zen3
module load file/5.46-gcc-14.2.0-nplp5fn
# libxml2@=2.15.3+pic~python+shared build_system=autotools platform=linux os=sles15 target=zen3
module load libxml2/2.15.3-gcc-14.2.0-wiekcrj
# pigz@=2.8 build_system=makefile platform=linux os=sles15 target=zen3
module load pigz/2.8-gcc-14.2.0-n72hmpe
# tar@=1.35 build_system=autotools zip=pigz platform=linux os=sles15 target=zen3
module load tar/1.35-gcc-14.2.0-fwjntwi
# gettext@=1.0+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools platform=linux os=sles15 target=zen3
module load gettext/1.0-gcc-14.2.0-tm7n7uv
# findutils@=4.10.0 build_system=autotools patches:=440b954 platform=linux os=sles15 target=zen3
module load findutils/4.10.0-gcc-14.2.0-4ob63oh
# libtool@=2.5.4 build_system=autotools platform=linux os=sles15 target=zen3
module load libtool/2.5.4-gcc-14.2.0-sqcpodk
# gmp@=6.3.0+cxx build_system=autotools libs:=shared,static platform=linux os=sles15 target=zen3
module load gmp/6.3.0-gcc-14.2.0-lhz367r
# autoconf-archive@=2024.10.16 build_system=autotools platform=linux os=sles15 target=zen3
module load autoconf-archive/2024.10.16-none-none-nwlbkhx
# texinfo@=7.2~xs build_system=autotools platform=linux os=sles15 target=zen3
module load texinfo/7.2-gcc-14.2.0-jsmo5ho
# mpfr@=4.2.2 build_system=autotools libs:=shared,static platform=linux os=sles15 target=zen3
module load mpfr/4.2.2-gcc-14.2.0-5asupl6
# suite-sparse@=7.12.2~cuda~graphblas~openmp+pic build_system=generic platform=linux os=sles15 target=zen3
module load suite-sparse/7.12.2-gcc-14.2.0-4s7azhu
# umpire@=2025.12.0~asan~backtrace+c~cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~mpi3_shmem~numa~omptarget~openmp+rocm~sanitizer_tests+shared~sqlite_experimental~tools~werror amdgpu_target:=gfx90a build_system=cmake build_type=Release commit=0372fbd6e1f17d7e6dd72693f8b857f3ec7559e9 generator=make tests=none platform=linux os=sles15 target=zen3
module load umpire/2025.12.0-gcc-14.2.0-abe4gxy
# hiop@=1.2.0~axom~cuda~deepchecking~ginkgo~ipo~jsrun+kron+mpi+raja+rocm~shared+sparse amdgpu_target:=gfx90a build_system=cmake build_type=Release commit=6acee835136e9beddf3570b28739a1b1001e528a generator=make patches:=e73daa7 platform=linux os=sles15 target=zen3
module load hiop/1.2.0-gcc-14.2.0-uulmsj5
# ipopt@=3.14.14+coinhsl~debug~java~metis~mumps build_system=autotools platform=linux os=sles15 target=zen3
module load ipopt/3.14.14-gcc-14.2.0-4227qwo
# parmetis@=4.0.3~gdb~int64~ipo+shared build_system=cmake build_type=Release generator=make patches:=4f89253,50ed208,704b84f platform=linux os=sles15 target=zen3
module load parmetis/4.0.3-gcc-14.2.0-kir7ich
# petsc@=3.25.2~X~batch~cgns~complex~cuda~debug+double+examples~exodusii~fftw+fortran+fortran-bindings~giflib~hdf5~hpddm~hwloc~hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~ml~mmg~moab~mpfr+mpi~mumps~openmp~p4est~parmmg~ptscotch~random123~rocm~saws~scalapack+shared~strumpack~suite-sparse~superlu-dist~sycl~tetgen~valgrind~zoltan build_system=generic clanguage=C memalign=none platform=linux os=sles15 target=zen3
module load petsc/3.25.2-gcc-14.2.0-sa3r4pp
# spdlog@=1.15.0~ipo+shared build_system=cmake build_type=Release cxxstd=14 generator=make patches:=5ed92f4,fd4cbb1,fdc325d platform=linux os=sles15 target=zen3
module load spdlog/1.15.0-gcc-14.2.0-chfsasm
# exago@=develop~cuda+hiop~ipo+ipopt+logging+mpi~python+raja+rocm+testing amdgpu_target:=gfx90a build_system=cmake build_type=Release dev_path=/lustre/orion/scratch/nkouk/stf006/Codes/ExaGO generator=make platform=linux os=sles15 target=zen3
## module load exago/develop-gcc-14.2.0-uk2jden

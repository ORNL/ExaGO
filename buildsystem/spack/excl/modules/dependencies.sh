module use -a /home/97k/Codes/ExaGO/.spack/spack-install/modules/linux-ubuntu24.04-zen3
module use -a /home/97k/Codes/ExaGO/.spack/spack-install/modules/linux-ubuntu24.04-x86_64
# compiler-wrapper@=1.1.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load compiler-wrapper/1.1.0-none-none-cj36drx
# gcc@=12.2.0+binutils+bootstrap~graphite+libsanitizer~mold~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' platform=linux os=ubuntu24.04 target=x86_64
module load gcc/12.2.0-none-none-zc7zy37
# glibc@=2.39 build_system=autotools platform=linux os=ubuntu24.04 target=x86_64
module load glibc/2.39-none-none-thy5vqo
# gcc-runtime@=12.2.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load gcc-runtime/12.2.0-none-none-spplgfu
# gmake@=4.4.1~guile build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load gmake/4.4.1-gcc-12.2.0-xkagqms
# libiconv@=1.18 build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load libiconv/1.18-gcc-12.2.0-ve5q5x4
# diffutils@=3.12 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load diffutils/3.12-gcc-12.2.0-jamk4jc
# pkgconf@=2.5.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load pkgconf/2.5.1-gcc-12.2.0-6hahqd7
# nghttp2@=1.67.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load nghttp2/1.67.1-gcc-12.2.0-ylzjz62
# ca-certificates-mozilla@=2026-03-19 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load ca-certificates-mozilla/2026-03-19-none-none-becq5hm
# berkeley-db@=18.1.40+cxx~docs+stl build_system=autotools patches:=26090f4,b231fcc platform=linux os=ubuntu24.04 target=zen3
module load berkeley-db/18.1.40-gcc-12.2.0-r3hdfby
# bzip2@=1.0.8~debug~pic+shared build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load bzip2/1.0.8-gcc-12.2.0-7mxenzw
# ncurses@=6.6~symlinks+termlib abi=none build_system=autotools patches:=7a351bc platform=linux os=ubuntu24.04 target=zen3
module load ncurses/6.6-gcc-12.2.0-cvjclb2
# readline@=8.3 build_system=autotools patches:=21f0a03,72dee13,e273643 platform=linux os=ubuntu24.04 target=zen3
module load readline/8.3-gcc-12.2.0-hrmgzzf
# gdbm@=1.26 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load gdbm/1.26-gcc-12.2.0-g42pm6w
# less@=692 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load less/692-gcc-12.2.0-vycuyql
# zlib-ng@=2.3.3+compat+new_strategies+opt+pic+shared build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load zlib-ng/2.3.3-gcc-12.2.0-y7jidfz
# perl@=5.42.0+cpanm+opcode+open+shared+threads build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load perl/5.42.0-gcc-12.2.0-piuagzy
# openssl@=3.6.1~docs+shared build_system=generic certs=mozilla platform=linux os=ubuntu24.04 target=zen3
## module load openssl/3.6.1-gcc-12.2.0-4ber6sf
# curl@=8.20.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs:=shared,static tls:=openssl platform=linux os=ubuntu24.04 target=zen3
module load curl/8.20.0-gcc-12.2.0-pxg7ava
# cmake@=3.31.11~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release platform=linux os=ubuntu24.04 target=zen3
module load cmake/3.31.11-gcc-12.2.0-wvogeze
# blt@=0.7.2 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load blt/0.7.2-gcc-12.2.0-lf4a734
# cuda@=12.0.140~allow-unsupported-compilers~dev build_system=generic platform=linux os=ubuntu24.04 target=x86_64
module load cuda/12.0.140-none-none-sl4gznh
# camp@=2025.12.0+cuda~ipo~omptarget~openmp~rocm~sycl~tests build_system=cmake build_type=RelWithDebInfo commit=a8caefa9f4c811b1a114b4ed2c9b681d40f12325 cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load camp/2025.12.0-gcc-12.2.0-os2ormt
# fmt@=11.0.2~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make platform=linux os=ubuntu24.04 target=zen3
module load fmt/11.0.2-gcc-12.2.0-w2burvi
# libmd@=1.1.0 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libmd/1.1.0-gcc-12.2.0-bevcujx
# libbsd@=0.12.2 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libbsd/0.12.2-gcc-12.2.0-wme4hwk
# expat@=2.8.1+libbsd build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load expat/2.8.1-gcc-12.2.0-wn73iyj
# xz@=5.8.3~pic build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load xz/5.8.3-gcc-12.2.0-aivzrr6
# libxml2@=2.15.3+pic~python+shared build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libxml2/2.15.3-gcc-12.2.0-rgajupn
# pigz@=2.8 build_system=makefile platform=linux os=ubuntu24.04 target=zen3
module load pigz/2.8-gcc-12.2.0-jpb4iod
# zstd@=1.5.7+programs build_system=makefile compression:=none libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load zstd/1.5.7-gcc-12.2.0-jztkoyq
# tar@=1.35 build_system=autotools zip=pigz platform=linux os=ubuntu24.04 target=zen3
module load tar/1.35-gcc-12.2.0-zbq2dgn
# gettext@=1.0+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load gettext/1.0-gcc-12.2.0-67vy5cs
# libffi@=3.5.2 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libffi/3.5.2-gcc-12.2.0-whtg5rk
# sqlite@=3.53.1+column_metadata+fts+rtree build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load sqlite/3.53.1-gcc-12.2.0-w5piscf
# util-linux-uuid@=2.41 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load util-linux-uuid/2.41-gcc-12.2.0-fzrtlth
# python@=3.14.5+bz2+ctypes+dbm~debug~freethreading+libxml2+lzma~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~static~tests~tkinter+uuid+zlib+zstd build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load python/3.14.5-gcc-12.2.0-2w22niw
# re2c@=4.4 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load re2c/4.4-gcc-12.2.0-gxf6t5a
# ninja@=1.13.2+re2c build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load ninja/1.13.2-gcc-12.2.0-5rqnmz3
# python-venv@=1.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load python-venv/1.0-none-none-7ay4m2c
# py-pip@=26.1.2 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-pip/26.1.2-none-none-mm3eevh
# py-setuptools@=82.0.1 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-setuptools/82.0.1-none-none-ghcleok
# py-wheel@=0.45.1 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-wheel/0.45.1-none-none-ap7qzct
# meson@=1.11.1 build_system=python_pip patches:=0f0b1bd platform=linux os=ubuntu24.04 target=zen3
module load meson/1.11.1-none-none-wvfsf4u
# metis@=5.1.0~gdb~int64~ipo~no_warning~real64+shared build_system=cmake build_type=Release generator=make patches:=4991da9,93a7903,b1225da platform=linux os=ubuntu24.04 target=zen3
module load metis/5.1.0-gcc-12.2.0-6bpv6xk
# openblas@=0.3.20~bignuma~consistent_fpcsr+dynamic_dispatch~ilp64+locking+pic+shared~static build_system=makefile patches:=9f12903 symbol_suffix=none threads=none platform=linux os=ubuntu24.04 target=zen3
module load openblas/0.3.20-gcc-12.2.0-q2yyi4a
# coinhsl@=2024.05.15+metis~strip build_system=meson buildtype=release default_library:=shared platform=linux os=ubuntu24.04 target=zen3
module load coinhsl/2024.05.15-gcc-12.2.0-qsqsec3
# magma@=2.8.0+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=RelWithDebInfo cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load magma/2.8.0-gcc-12.2.0-gjleh3e
# openmpi@=5.10.0+atomics~cuda~debug+fortran~gpfs~internal-hwloc~internal-libevent~internal-pmix~ipv6~java~lustre~memchecker~openshmem~rocm+romio+rsh~static~two_level_namespace+vt+wrapper-rpath build_system=autotools fabrics:=none romio-filesystem:=none schedulers:=none platform=linux os=ubuntu24.04 target=x86_64
module load openmpi/5.10.0-none-none-o7fyiig
# raja@=2025.12.2~caliper+cuda~desul~examples~exercises~gpu-profiling~ipo~lowopttest~omptarget~omptask~openmp~plugins~rocm~run-all-tests~shared~sycl~tests+vectorization build_system=cmake build_type=RelWithDebInfo commit=eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274 cuda_arch:=70 cxxstd=20 generator=make platform=linux os=ubuntu24.04 target=zen3
module load raja/2025.12.2-gcc-12.2.0-jx7hdmf
# libsigsegv@=2.15 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libsigsegv/2.15-gcc-12.2.0-myadrl7
# m4@=1.4.21+sigsegv build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load m4/1.4.21-gcc-12.2.0-avwyo7g
# autoconf@=2.72 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load autoconf/2.72-none-none-n773lt6
# automake@=1.18.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load automake/1.18.1-gcc-12.2.0-tacfeac
# file@=5.46+static build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load file/5.46-gcc-12.2.0-j4mp2kz
# findutils@=4.10.0 build_system=autotools patches:=440b954 platform=linux os=ubuntu24.04 target=zen3
module load findutils/4.10.0-gcc-12.2.0-efx4n5n
# libtool@=2.5.4 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libtool/2.5.4-gcc-12.2.0-t6wofyx
# gmp@=6.3.0+cxx build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load gmp/6.3.0-gcc-12.2.0-zdhq5rd
# autoconf-archive@=2024.10.16 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load autoconf-archive/2024.10.16-none-none-nwrirxb
# texinfo@=7.2~xs build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load texinfo/7.2-gcc-12.2.0-zpbklfi
# mpfr@=4.2.2 build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load mpfr/4.2.2-gcc-12.2.0-ikbj5pn
# suite-sparse@=7.12.2~cuda~graphblas~openmp+pic build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load suite-sparse/7.12.2-gcc-12.2.0-ior3blk
# umpire@=2025.12.0~asan~backtrace+c+cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~mpi3_shmem~numa~omptarget~openmp~rocm~sanitizer_tests+shared~sqlite_experimental~tools~werror build_system=cmake build_type=RelWithDebInfo commit=0372fbd6e1f17d7e6dd72693f8b857f3ec7559e9 cuda_arch:=70 generator=make tests=none platform=linux os=ubuntu24.04 target=zen3
module load umpire/2025.12.0-gcc-12.2.0-hdgwsdb
# hiop@=1.1.1~axom+cuda+cusolver_lu+deepchecking~ginkgo~ipo~jsrun+kron+mpi+raja~rocm~shared+sparse build_system=cmake build_type=RelWithDebInfo commit=d8762e05150b2040a27f69d8bf6603f22190a869 cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load hiop/1.1.1-gcc-12.2.0-t2sc4rj
# ipopt@=3.14.14+coinhsl~debug~java~metis~mumps build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load ipopt/3.14.14-gcc-12.2.0-ou426fz
# parmetis@=4.0.3~gdb~int64~ipo+shared build_system=cmake build_type=Release generator=make patches:=4f89253,50ed208,704b84f platform=linux os=ubuntu24.04 target=zen3
module load parmetis/4.0.3-gcc-12.2.0-ydtm25v
# petsc@=3.25.2~X~batch~cgns~complex~cuda~debug+double+examples~exodusii~fftw+fortran+fortran-bindings~giflib~hdf5~hpddm~hwloc~hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~ml~mmg~moab~mpfr+mpi~mumps~openmp~p4est~parmmg~ptscotch~random123~rocm~saws~scalapack+shared~strumpack~suite-sparse~superlu-dist~sycl~tetgen~valgrind~zoltan build_system=generic clanguage=C memalign=none platform=linux os=ubuntu24.04 target=zen3
module load petsc/3.25.2-gcc-12.2.0-2xyumfa
# spdlog@=1.15.0~ipo+shared build_system=cmake build_type=Release cxxstd=14 generator=make patches:=5ed92f4,fd4cbb1,fdc325d platform=linux os=ubuntu24.04 target=zen3
module load spdlog/1.15.0-gcc-12.2.0-uuuftco
# exago@=develop+cuda+hiop~ipo+ipopt+logging+mpi~python+raja~rocm+testing build_system=cmake build_type=RelWithDebInfo cuda_arch:=70 dev_path=/home/97k/Codes/ExaGO generator=make platform=linux os=ubuntu24.04 target=zen3
## module load exago/develop-gcc-12.2.0-l6igc2x

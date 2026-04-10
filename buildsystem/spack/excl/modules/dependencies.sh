module use -a /home/97k/Codes/ExaGO/tpl/spack/share/spack/modules/linux-ubuntu24.04-zen3
module use -a /home/97k/Codes/ExaGO/tpl/spack/share/spack/modules/linux-ubuntu24.04-x86_64
# compiler-wrapper@=1.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load compiler-wrapper/1.0-none-none-x2gzyyl
# gcc@=12.2.0~binutils+bootstrap~graphite~mold~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' platform=linux os=ubuntu24.04 target=x86_64
module load gcc/12.2.0-none-none-wwvcn4f
# glibc@=2.39 build_system=autotools platform=linux os=ubuntu24.04 target=x86_64
module load glibc/2.39-none-none-thy5vqo
# gcc-runtime@=12.2.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load gcc-runtime/12.2.0-none-none-wy5ooxl
# gmake@=4.4.1~guile build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load gmake/4.4.1-gcc-12.2.0-efctz62
# libiconv@=1.18 build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load libiconv/1.18-gcc-12.2.0-i7p4nbp
# diffutils@=3.12 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load diffutils/3.12-gcc-12.2.0-je24hvc
# pkgconf@=2.5.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load pkgconf/2.5.1-gcc-12.2.0-t4pnt3x
# nghttp2@=1.67.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load nghttp2/1.67.1-gcc-12.2.0-dq3ao4v
# ca-certificates-mozilla@=2025-08-12 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load ca-certificates-mozilla/2025-08-12-none-none-vva5gkb
# berkeley-db@=18.1.40+cxx~docs+stl build_system=autotools patches:=26090f4,b231fcc platform=linux os=ubuntu24.04 target=zen3
module load berkeley-db/18.1.40-gcc-12.2.0-ljrgpwi
# bzip2@=1.0.8~debug~pic+shared build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load bzip2/1.0.8-gcc-12.2.0-akcgybl
# ncurses@=6.5-20250705~symlinks+termlib abi=none build_system=autotools patches:=7a351bc platform=linux os=ubuntu24.04 target=zen3
module load ncurses/6.5-20250705-gcc-12.2.0-ywbssmz
# readline@=8.3 build_system=autotools patches:=21f0a03 platform=linux os=ubuntu24.04 target=zen3
module load readline/8.3-gcc-12.2.0-hkopiw7
# gdbm@=1.25 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load gdbm/1.25-gcc-12.2.0-lczy4om
# zlib-ng@=2.2.4+compat+new_strategies+opt+pic+shared build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load zlib-ng/2.2.4-gcc-12.2.0-jex3xzg
# perl@=5.42.0+cpanm+opcode+open+shared+threads build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load perl/5.42.0-gcc-12.2.0-zs56xud
# openssl@=3.6.0~docs+shared build_system=generic certs=mozilla platform=linux os=ubuntu24.04 target=zen3
## module load openssl/3.6.0-gcc-12.2.0-zpjdgj7
# curl@=8.15.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs:=shared,static tls:=openssl platform=linux os=ubuntu24.04 target=zen3
module load curl/8.15.0-gcc-12.2.0-edykktr
# cmake@=3.31.9~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release platform=linux os=ubuntu24.04 target=zen3
module load cmake/3.31.9-gcc-12.2.0-7gyw2jj
# blt@=0.7.1 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load blt/0.7.1-gcc-12.2.0-2g34kmy
# cuda@=12.0.140~allow-unsupported-compilers~dev build_system=generic platform=linux os=ubuntu24.04 target=x86_64
module load cuda/12.0.140-none-none-6a7oxtn
# camp@=2025.03.0+cuda~ipo~omptarget~openmp~rocm~sycl~tests build_system=cmake build_type=Release commit=ee0a3069a7ae72da8bcea63c06260fad34901d43 cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load camp/2025.03.0-gcc-12.2.0-csq277i
# fmt@=11.0.2~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make platform=linux os=ubuntu24.04 target=zen3
module load fmt/11.0.2-gcc-12.2.0-7pcqvtx
# libmd@=1.1.0 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libmd/1.1.0-gcc-12.2.0-urxul4u
# libbsd@=0.12.2 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libbsd/0.12.2-gcc-12.2.0-u7r6c2e
# expat@=2.7.3+libbsd build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load expat/2.7.3-gcc-12.2.0-bkpwnm2
# xz@=5.6.3~pic build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load xz/5.6.3-gcc-12.2.0-kal3v7p
# libxml2@=2.13.5~http+pic~python+shared build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libxml2/2.13.5-gcc-12.2.0-t24wuem
# pigz@=2.8 build_system=makefile platform=linux os=ubuntu24.04 target=zen3
module load pigz/2.8-gcc-12.2.0-xqgxraf
# zstd@=1.5.7+programs build_system=makefile compression:=none libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load zstd/1.5.7-gcc-12.2.0-ycvkiar
# tar@=1.35 build_system=autotools zip=pigz platform=linux os=ubuntu24.04 target=zen3
module load tar/1.35-gcc-12.2.0-kfwxzvj
# gettext@=0.23.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load gettext/0.23.1-gcc-12.2.0-fpujsha
# libffi@=3.5.2 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libffi/3.5.2-gcc-12.2.0-iim37gt
# sqlite@=3.50.4+column_metadata+fts+rtree build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load sqlite/3.50.4-gcc-12.2.0-jdatat5
# util-linux-uuid@=2.41 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load util-linux-uuid/2.41-gcc-12.2.0-66smhv2
# python@=3.14.0+bz2+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib+zstd build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load python/3.14.0-gcc-12.2.0-q3qvdlk
# re2c@=3.1 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load re2c/3.1-gcc-12.2.0-ca6glii
# ninja@=1.13.0+re2c build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load ninja/1.13.0-gcc-12.2.0-qkkutua
# python-venv@=1.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load python-venv/1.0-none-none-jpvakyo
# py-pip@=25.1.1 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-pip/25.1.1-none-none-hogxyko
# py-setuptools@=80.9.0 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-setuptools/80.9.0-none-none-lhxdf6r
# py-wheel@=0.45.1 build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load py-wheel/0.45.1-none-none-pziyirk
# meson@=1.8.5 build_system=python_pip patches:=0f0b1bd platform=linux os=ubuntu24.04 target=zen3
module load meson/1.8.5-none-none-dh4ftn6
# metis@=5.1.0~gdb~int64~ipo~no_warning~real64+shared build_system=cmake build_type=Release generator=make patches:=4991da9,93a7903,b1225da platform=linux os=ubuntu24.04 target=zen3
module load metis/5.1.0-gcc-12.2.0-pytmuow
# openblas@=0.3.20~bignuma~consistent_fpcsr+dynamic_dispatch~ilp64+locking+pic+shared build_system=makefile patches:=9f12903 symbol_suffix=none threads=none platform=linux os=ubuntu24.04 target=zen3
module load openblas/0.3.20-gcc-12.2.0-knvevhk
# coinhsl@=2024.05.15+metis~strip build_system=meson buildtype=release default_library:=shared platform=linux os=ubuntu24.04 target=zen3
module load coinhsl/2024.05.15-gcc-12.2.0-yc7hx4y
# magma@=2.8.0+cuda+fortran~ipo~rocm+shared build_system=cmake build_type=Release cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load magma/2.8.0-gcc-12.2.0-zn4unow
# openmpi@=5.0.1+atomics~cuda~debug+fortran~gpfs~internal-hwloc~internal-libevent~internal-pmix~ipv6~java~lustre~memchecker~openshmem~rocm+romio+rsh~static~two_level_namespace+vt+wrapper-rpath build_system=autotools fabrics:=none romio-filesystem:=none schedulers:=none platform=linux os=ubuntu24.04 target=x86_64
module load openmpi/5.0.1-none-none-luag4ys
# raja@=2025.03.0+cuda~desul~examples~exercises~gpu-profiling~ipo~lowopttest~omptarget~omptask~openmp~plugins~rocm~run-all-tests~shared~sycl~tests~vectorization build_system=cmake build_type=Release commit=1d70abf171474d331f1409908bdf1b1c3fe19222 cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load raja/2025.03.0-gcc-12.2.0-4r2ytzu
# libsigsegv@=2.14 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libsigsegv/2.14-gcc-12.2.0-lg7jej4
# m4@=1.4.20+sigsegv build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load m4/1.4.20-gcc-12.2.0-upaifnu
# autoconf@=2.72 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load autoconf/2.72-none-none-5r6lwbo
# automake@=1.16.5 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load automake/1.16.5-gcc-12.2.0-m5esjyc
# findutils@=4.10.0 build_system=autotools patches:=440b954 platform=linux os=ubuntu24.04 target=zen3
module load findutils/4.10.0-gcc-12.2.0-p6e75f4
# libtool@=2.4.7 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load libtool/2.4.7-gcc-12.2.0-rqagg4t
# gmp@=6.3.0+cxx build_system=autotools libs:=shared,static platform=linux os=ubuntu24.04 target=zen3
module load gmp/6.3.0-gcc-12.2.0-qvldnwt
# autoconf-archive@=2023.02.20 build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load autoconf-archive/2023.02.20-none-none-iwglul3
# texinfo@=7.2~xs build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load texinfo/7.2-gcc-12.2.0-ndhtb6p
# mpfr@=4.2.1 build_system=autotools libs:=shared,static patches:=3ec29a6 platform=linux os=ubuntu24.04 target=zen3
module load mpfr/4.2.1-gcc-12.2.0-kk2qthy
# suite-sparse@=7.8.3~cuda~graphblas~openmp+pic build_system=generic platform=linux os=ubuntu24.04 target=zen3
module load suite-sparse/7.8.3-gcc-12.2.0-ulqsjxz
# umpire@=2025.03.0~asan~backtrace+c+cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~mpi3_shmem~numa~omptarget~openmp~rocm~sanitizer_tests~shared~sqlite_experimental~tools~werror build_system=cmake build_type=Release commit=1ed0669c57f041baa1f1070693991c3a7a43e7ee cuda_arch:=70 generator=make tests=none platform=linux os=ubuntu24.04 target=zen3
module load umpire/2025.03.0-gcc-12.2.0-uluaqpx
# hiop@=1.1.1~axom+cuda+cusolver_lu~deepchecking~ginkgo~ipo~jsrun+kron+mpi+raja~rocm~shared+sparse build_system=cmake build_type=RelWithDebInfo commit=d8762e05150b2040a27f69d8bf6603f22190a869 cuda_arch:=70 generator=make platform=linux os=ubuntu24.04 target=zen3
module load hiop/1.1.1-gcc-12.2.0-2hox7q4
# ipopt@=3.14.14+coinhsl~debug~java~metis~mumps build_system=autotools platform=linux os=ubuntu24.04 target=zen3
module load ipopt/3.14.14-gcc-12.2.0-wkqcozn
# parmetis@=4.0.3~gdb~int64~ipo+shared build_system=cmake build_type=Release generator=make patches:=4f89253,50ed208,704b84f platform=linux os=ubuntu24.04 target=zen3
module load parmetis/4.0.3-gcc-12.2.0-w3wzpzx
# petsc@=3.24.1~X~batch~cgns~complex~cuda~debug+double+examples~exodusii~fftw+fortran+fortran-bindings~giflib~hdf5~hpddm~hwloc~hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi~mumps~openmp~p4est~parmmg~ptscotch~random123~rocm~saws~scalapack+shared~strumpack~suite-sparse~superlu-dist~sycl~tetgen~trilinos~valgrind~zoltan build_system=generic clanguage=C memalign=none patches:=fa5ef56 platform=linux os=ubuntu24.04 target=zen3
module load petsc/3.24.1-gcc-12.2.0-suz6dyi
# spdlog@=1.15.0~ipo+shared build_system=cmake build_type=Release generator=make patches:=5ed92f4,fd4cbb1,fdc325d platform=linux os=ubuntu24.04 target=zen3
module load spdlog/1.15.0-gcc-12.2.0-buhc4zb
# exago@=develop+cuda+hiop~ipo+ipopt+logging+mpi~python+raja~rocm+testing build_system=cmake build_type=MinSizeRel cuda_arch:=70 dev_path=/home/97k/Codes/ExaGO generator=make platform=linux os=ubuntu24.04 target=zen3
## module load exago/develop-gcc-12.2.0-nzclj3v

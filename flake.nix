{
  description = "ExaGO development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hiop-src = {
      url = "github:LLNL/hiop/v1.2.0";
      flake = false;
    };
  };

  outputs =
    { hiop-src, nixpkgs, ... }:
    let
      lib = nixpkgs.lib;
      supportedSystems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
    in
    {
      devShells = lib.genAttrs supportedSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          # don't use ilp64 as this conflicts with petsc and ipopt
          openblas = pkgs.openblas.override { blas64 = false; };
          # ipopt's sequential mumps exports fake mpi symbols by default which
          # conflict with exago's real mpi
          mumpsMpi = (pkgs.mumps.override { mpiSupport = true; }).overrideAttrs (prev: {
            postInstall = (prev.postInstall or "") + ''
              ln -s . "$out/include/mumps_seq"
            '';
            passthru = (prev.passthru or { }) // { mpiSupport = false; };
          });
          ipopt = (pkgs.ipopt.override { mumps = mumpsMpi; }).overrideAttrs (prev: {
            buildInputs = (prev.buildInputs or [ ]) ++ [ pkgs.mpi ];
          });
          hiop = pkgs.stdenv.mkDerivation {
            pname = "hiop";
            version = "1.2.0";
            src = hiop-src;
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
              pkgs.gfortran
              pkgs.pkg-config
            ];

            buildInputs = [
              openblas
              pkgs.eigen
              pkgs.metis
              pkgs.mpi
            ];

            cmakeFlags = [
              "-GNinja"
              "-DHIOP_BUILD_SHARED=ON"
              "-DHIOP_BUILD_STATIC=OFF"
              "-DHIOP_USE_MPI=ON"
              "-DHIOP_USE_GPU=OFF"
              "-DHIOP_USE_RAJA=OFF"
              "-DHIOP_SPARSE=ON"
              "-DHIOP_USE_COINHSL=ON"
              "-DHIOP_USE_GINKGO=OFF"
              "-DHIOP_USE_PARDISO=OFF"
              "-DHIOP_USE_STRUMPACK=OFF"
              "-DHIOP_METIS_DIR=${pkgs.metis}"
              "-DHIOP_EIGEN_DIR=${pkgs.eigen}"
              "-DHIOP_WITH_MAKETEST=OFF"
              "-DHIOP_BUILD_DOCUMENTATION=OFF"
              "-DHIOP_BUILD_FORTRAN_EXAMPLE=OFF"
            ];
            # exago wants this adapter which hiop will only install when drivers
            # are enabled. we install it without building them
            postInstall = ''
              install -Dm444 "$src/src/Drivers/IpoptAdapter/IpoptAdapter.hpp" \
                "$out/include/IpoptAdapter.hpp"
            '';
          };
        in
        {
          default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
              pkgs.pkg-config
              pkgs.perl
            ];

            buildInputs = [
              openblas
              ipopt
              pkgs.mpi
              pkgs.spdlog
              pkgs.fmt
              pkgs.ruff
              hiop
              pkgs.petsc
            ]
            ++ lib.attrVals [
              "python"
              "flask"
              "flask-cors"
              "langchain"
              "langchain-anthropic"
              "langchain-openai"
              "langchain-ollama"
              "streamlit"
              "plotly"
              "reportlab"
              "shapely"
              "pandas"
              "geopandas"
              "duckdb"
              "mpi4py"
              "pytest"
            ] pkgs.python314Packages;

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
            ];
          };
        }
      );
    };
}

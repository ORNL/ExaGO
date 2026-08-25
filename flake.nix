{
  description = "ExaGO development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      lib = nixpkgs.lib;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
        ];

        buildInputs = [
          pkgs.blas
          pkgs.lapack
          pkgs.ipopt
          pkgs.mpi
          pkgs.petsc
          pkgs.spdlog
          pkgs.fmt
          pkgs.ruff
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
    };
}

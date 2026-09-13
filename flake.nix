{
  description = "Chonks — local code RAG with MCP server for Claude Code";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python312;

        nativeDeps = with pkgs; [
          stdenv.cc.cc.lib
          zlib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            uv

            nodejs_20

            sqlite
            sqlite.dev

            git
            jq
            ripgrep
          ] ++ nativeDeps;

          shellHook = ''
            export UV_PYTHON="${python}/bin/python"
            export UV_PYTHON_DOWNLOADS=never

            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeDeps}:''${LD_LIBRARY_PATH:-}"

            export PYTHONNOUSERSITE=1

            if [ ! -d .venv ]; then
              echo "[flake] no .venv yet — run 'uv sync' to create one"
            fi

            echo "[flake] python: $(python --version 2>&1)"
            echo "[flake] uv:     $(uv --version 2>&1)"
            echo "[flake] node:   $(node --version 2>&1)"
          '';
        };
      });
}

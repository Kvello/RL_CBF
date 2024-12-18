{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.11-pre.tar.gz") {} }:
let
  python = pkgs.python311;
in
pkgs.mkShellNoCC {
  packages = with pkgs; [
    (python.withPackages (ps: with ps; [
      numpy
      scipy
      jupyter
      pyqt6
      ipython
      matplotlib
      pytest
      flake8
      pylint
      autopep8
      tqdm
      pywayland
      pandas
      seaborn
      tqdm
    ]))
  ];
  shellHook = ''
  '';
}

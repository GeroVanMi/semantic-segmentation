let
  nixpkgs_unstable = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-unstable";
  pkgs_unstable = import nixpkgs_unstable { config = {}; overlays = []; };
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in


pkgs.mkShellNoCC {
  packages = [
    pkgs.python311Full
    pkgs.python311Packages.wandb
    pkgs.python311Packages.pandas
    pkgs_unstable.python311Packages.torch
    pkgs_unstable.python311Packages.torchvision
  ];
  shellHook = ''
    fish
  '';
}



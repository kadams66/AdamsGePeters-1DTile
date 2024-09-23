# AdamsGePeters-1DTile
Data and code used in our 1D tiling paper

## kMC Data
This folder contains each of the kMC datasets used in constructing all the plots in the main text and SI of the paper. Each .dat file is a plaintext data file with the kMC computed averages at each parameter set, as denoted by the header in the file. The params.json contains all the parameters used in the corresponding kMC run

## Tile.py
The primary code for calculating the 1D tiling using a set of parameters. Split into three parts. The first is a generic `Reactions` object for use in modeling all the reactions that can occur within the tile/system. The second is the `Brickwork` 1D Tile object that uses a `Reactions` object and a specified `l` and `d` to calculate the reaction or coverage vectors ($` r_{P}[x] `$ or $` O_{i}[x] `$ in the paper) and steady-state thetas ($` \Theta_{ss} `$ in the paper) for a given set of elementary reaction rates. The last is a set of example `Reactions` objects for specific systems, both those explored in the main text and SI but also some additional systems. 

## Tile_Data.py
A helper script that uses the params.json of a specified kMC run to generate the corresponding 1D tiling data for that run using Tile.py. Interfaces with Tile_params.json to determine which kMC dataset, `l`, `d`, and the resolution of the corresponding tiling data to use.

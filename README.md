# MosaicADAPT-QAOA-Experiments

This library consumes the fork of [MosaicADAPT-QAOA](https://github.com/pratimugale/MosaicADAPT-QAOA-Experiments) and contains the source code for the experiments that test the performance of MosaicADAPT-QAOA, Tetris-QAOA and ADAPT-QAOA on random 3-SAT instances.
The original MosaicADAPT-QAOA repository contains the implementation of the algorithm, while this repository contains the source code for the experiments on top of the variants in that repository.

## Installation

1. Clone this repo:
```bash
git clone https://github.com/pratimugale/MosaicADAPT-QAOA-Experiments
```
2. Run the installation script: `bash install.sh` which clones the dependent MosaicADAPT-QAOA repo and creates the Python virtual environment with the required packages, and sets up the Julia package.
3. Activate the environment: `source venv/bin/activate`
4. Install MosaicADAPT-QAOA using `cd MosaicADAPT-QAOA && make install` (see [MosaicADAPT-QAOA/README.md](MosaicADAPT-QAOA/README.md)). Run a smoke test using `make smoke`.
5. To use MosaicADAPT-QAOA, we additionally need to build the KaMIS binary. This can be done using `cd MosaicADAPT-QAOA && make install-kamis` (see [MosaicADAPT-QAOA/README.md](MosaicADAPT-QAOA/README.md)), which clones the KaMIS repo at the required relative path `external/KaMIS`. Note that the KaMIS binary needs to be built manually depending on the CPU architecture (Mac/Windows/Linux). Please follow the instructions at [MosaicADAPT-QAOA/external/KaMIS/HowToMacOs.md](MosaicADAPT-QAOA/external/KaMIS/HowToMacOs.md) if using MacOS. You might also need to edit `MosaicADAPT-QAOA/external/KaMIS/compile_withcmake.sh` and `MosaicADAPT-QAOA/external/KaMIS/mmwis/compile.sh` to use your installed version / path of gcc. To build KaMIS mmwis, you might likely need to run `cd external/KaMIS && ./compile_withcmake.sh`. Note that we only need the `mmwis` binary to be built for our usage.
6. Run `make smoke-kamis` to verify if the installation of mmwis has successfully completed. This completes the installation of the repo.

## Usage
1. Generate the dataset containing Max3SAT instances - 50 `uniform random` and 50 `balanced` using: `bash scripts/generate_dataset.sh <NUM_VARS>`.
2. Set the initial gamma that needs to be tested in [experiments/qaoa-sat.jl](experiments/qaoa-sat.jl) in the `gammas` variable.
3. Run the ADAPT variants - MosaicADAPT-QAOA, Tetris-QAOA and ADAPT-QAOA - for using the script provided at [scripts/run_mosaic_experiment.sh](scripts/run_mosaic_experiment.sh). The script will also generate the plots that compare the performance of the 3 methods. 

Note: This repo was called mis-tetris-adapt earlier, with the Julia package being called "MIS_TETRIS_ADAPT.jl". MIS stands for Max Independent Set, which is what we solve in MosaicADAPT-QAOA. Whenever we refer to the "greedy" method in this repo, we refer to TETRIS-QAOA. Whenever we refer to "kamis" or "MIS" method, we refer to MosaicADAPT-QAOA. This renaming will be done soon in the future.

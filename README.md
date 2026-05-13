# MosaicADAPT-QAOA-Experiments

This library consumes the fork of [MosaicADAPT-QAOA](https://github.com/pratimugale/MosaicADAPT-QAOA) and contains the source code for the experiments that test the performance of MosaicADAPT-QAOA, Tetris-QAOA and ADAPT-QAOA on random 3-SAT instances.
The original MosaicADAPT-QAOA repository contains the implementation of the algorithm, while this repository contains the source code for the experiments on top of the variants in that repository.

## Installation

1. Clone this repo with the submodule:
```bash
git clone --recurse-submodules https://github.com/pratimugale/mis-tetris-adapt.git
```
2. Install MosaicADAPT-QAOA. See the instructions in the MosaicADAPT-QAOA repository.
3. Generate the dataset containing Max3SAT instances - half `uniform random` and half `balanced`, and run the ADAPT variants - MosaicADAPT-QAOA, Tetris-QAOA and ADAPT-QAOA - for  using the slurm script provided at [scripts/run_convergence_analysis_across_qaoa_methods.sh](scripts/run_convergence_analysis_across_qaoa_methods.sh). The script will also generate the plots that compare the performance of the 3 methods. 

Note: This repo was called mis-tetris-adapt earlier, with the Julia package being called "MIS_TETRIS_ADAPT.jl". MIS stands for Max Independent Set, which is what we solve in MosaicADAPT-QAOA. Whenever we refer to the "greedy" method in this repo, we refer to TETRIS-QAOA. Whenever we refer to "kamis" or "MIS" method, we refer to MosaicADAPT-QAOA. This renaming will be done soon in the future.
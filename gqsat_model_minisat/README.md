# GraphQSAT C++ implementation using LibTorch

## Building

In order for the project to build and execute successfully, these packages are required to be installed on your system:
- G++ compiler (any recent version should suffice)
- CUDA Toolkit (version 11.3 or any newer 11.x version)
- cuDNN (version 8.7.0)

Then proceed as follows:
- head to [the PyTorch repository on GitHub](https://github.com/pytorch/pytorch) and clone it next to the root directory of the ML4SAT repository
- execute `git checkout v1.12.1` in order to set back to the PyTorch version used in the project
- create a Conda environment and install required packages by running
  ```
  conda install cmake ninja mkl mkl-include
  conda install -c pytorch magma-cuda11x
  pip install -r <pytorch-dir>/requirements.txt
  ```
  **NOTE:** Here `11x` denotes the version of CUDA installed on your system and `<pytorch-dir>` denotes the root directory of cloned pytorch repository
- execute `mkdir build_libtorch && cd build_libtorch && python ../tools/build_libtorch.py` with the recently created environment activated (this script is responsible for building the pytorch C++ distribution of the required version from source, so the execution may take a while)
- head back to this directory and execute `export MROOT=$(pwd) && cd simp && gmake r` (exported variable will invalidate after the terminal is closed, so if you wish to keep it, you may add `export MROOT=<absolute-path-to-this-dir>` to your `.bash_profile` or a similar file for your shell)
- `minisat_release` exectuable should appear in the `simp` directory

*P.S. If you choose to perform inference on a GPU, a model that supports version 11.3 or newer of CUDA is required and the corresponding drivers should be installed.*

## Abstract from the original minisat README

```
================================================================================
DIRECTORY OVERVIEW:

mtl/            Mini Template Library
utils/          Generic helper code (I/O, Parsing, CPU-time, etc)
core/           A core version of the solver
simp/           An extended solver with simplification capabilities
README
LICENSE

================================================================================
EXAMPLES:

Run minisat with same heuristics as version 2.0:

> minisat <cnf-file> -no-luby -rinc=1.5 -phase-saving=0 -rnd-freq=0.02
```

# Single GPU to Multi-GPU


### Introduction

In this project, we measured multiple algorithms and models to obtain indicator data to study when single GPU or multiple GPUs is better.

### Algorithms & models

1. **mergesort.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- compile: `nvcc -o mergesort mergesort.cu`
- execute: `./mergesort n who`
  - n = size of int array to be sorte
  - who = 0: sequential version on CPU, 1: Single GPU version, 2: Multi-GPU version

2. **montecarlo.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- compile: `nvcc -o montecarlo montecarlo.cu`
- execute: `./montecarlo n who`
  - n = number of iterations
  - who = 0: sequential version on CPU, 1: Single GPU version, 2: Multi-GPU version

3. **Pytorch-MNIST**
- Host: cuda3.cims.nyu.edu
- CUDA Version `module load cuda-11.8`
- execute:
  - [mnist/run_app.sh](mnist/run_app.sh): Test all diff dataset size for single & multi, get the running time
  - [mnist/run_ncu.sh](mnist/run_ncu.sh): Use NCU test all diff dataset size for single GPU version, get FLOPs and DRAM
  - Detail see [mnist/README.md](mnist/README.md)

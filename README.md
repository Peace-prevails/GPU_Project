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
  - Set Python ENV
    - If you are in cuda3.cims.nyu.edu. Just `export PATH="/scratch/hl5262/miniconda3/bin/:$PATH"`
    - If you test on other host, create a env with Python=3.11 and install pytorch with `conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
  - `cd mnist` go to the **mnist** dir, then run the Script
  - [run_app.sh](mnist/run_app.sh): Test all diff dataset size for single & multi, get the running time
  - [run_ncu.sh](mnist/run_ncu.sh): Use NCU test all diff dataset size for single GPU version, get FLOPs and DRAM
  - Detail see [mnist/README.md](mnist/README.md)


4. **knn_single_gpu.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- compile: `nvcc -o knn_single knn_single_gpu.cu`
- execute: `./knn_single synthetic_knn_dataset.csv `

5. **knn_dual_gpu.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- compile: `nvcc -o knn_dual knn_dual_gpu.cu`
- execute: `./knn_dual synthetic_knn_dataset.csv`

6. **single_cnn.py**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- execute: `python3 single_cnn.py`


7. **mul_cnn.py**
- Host: cuda4.cims.nyu.edu
- CUDA Version: `module load cuda-11.4`
- execute: `python3 -m torch.distributed.launch mul_cnn.py`

8. **nbody.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: `module load cuda-11.8`
- compile: nvcc -o nbody nbody.cu
- execute: `./nbody 10000 1`
- N represents the size of the data, and type indicates the execution mode (1 for single GPU, 2 for multi-GPU).

9. **reduce.cu**
- Host: cuda3.cims.nyu.edu
- CUDA Version: ` nvcc -o reduce reduce.cu`
- CUDA Version: `module load cuda-11.8`
- execute: ./reduce 10000 1`
- N represents the size of the data, and type indicates the execution mode (1 for single GPU, 2 for multi-GPU).

10. **GPU Final Project.ipynb**
- This jupyter notebook file includes how we prepared the data and trained the model.

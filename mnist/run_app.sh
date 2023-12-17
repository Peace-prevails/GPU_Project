#!/bin/bash

# List of matrix dimensions
dataset_size=(8192 32768 65536 131072 )

# Run the program for each dimension
for dim in "${dataset_size[@]}"; do
    echo "Running for dataset_size: $dim"

    python single_gpu.py  10 0 --dataset_size $dim > ./run_outputs/${dim}_singlegpu.log
    echo

    python multigpu.py  10 0 --dataset_size $dim > ./run_outputs/${dim}_multigpu.log
    echo

done

echo "All tests completed."
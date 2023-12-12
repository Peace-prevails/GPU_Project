#!/bin/bash

# List of matrix dimensions
dataset_size=(8192 32768 65536 131072 )

# Run the program for each dimension
for dim in "${dataset_size[@]}"; do
    echo "Running NCU for dataset_size: $dim"
    
    # get FLOPs
    echo "-- Get FLOPs --"
    ncu --profile-from-start off --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum --target-processes all  python single_gpu.py 1 1 > run_outputs/ncu/ncu_${dim}_FLOPs.log
    echo

    # get Global Mem
    echo "-- Get DRAM --"
    ncu --profile-from-start off --metrics dram_read_bytes,dram_write_bytes --target-processes all  python single_gpu.py 1 1 > run_outputs/ncu/ncu_${dim}_dram.log
    echo

done

echo "All tests completed."

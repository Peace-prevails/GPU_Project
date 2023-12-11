# Pytorch-MAIST

This is a pytorch version MAIST model.

Single GPU and multiple GPUs were used to test performance data.

## Files

* [my_model.py](my_model.py): A MAIST model with 6 layers

* [single_gpu.py](single_gpu.py): Non-distributed training script, use single GPU

* [multigpu.py](multigpu.py): DDP on a single node, use multi GPUs on the node

* [datautils.py](datautils.py): Generate dataset class

* [run_all.sh](run_all.sh): Test all diff dataset size for single & multi





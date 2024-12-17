#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

export TVM_HOME=/eos/user/m/mimodekj/env/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

export TF_ENABLE_ONEDNN_OPTS=0
date
mkdir benchmarks
mkdir output
mkdir code
mkdir onnx_benchmark

ls /eos/user/m/mimodekj/tensorflow_datasets

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

nvidia-smi
lscpu
date

#python3 benchmark.py --batch 4096 --network mlp_single --cluster $1 --code_generation --target opencl --amount 819200 --old_cluster 7225626

#date
#python3 benchmark.py --batch 4096 --network mlp_single --cluster $1 --code_generation --target cuda --amount 819200 --old_cluster 7225626
#date
#python3 benchmark.py --batch 4096 --network mlp_single --cluster $1 --code_generation --target llvm --amount 819200 --old_cluster 7225626
#date
#cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
#cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
#cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
#date
python3 convert2onnx.py --windowsize 3 --old_cluster 11
python3 benchmark.py --batch 16384 --network mlp_single --cluster $1 --amount 200 --old_cluster 11 --version 0.0.0 --target cuda

#cp -r onnx_benchmark/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

date

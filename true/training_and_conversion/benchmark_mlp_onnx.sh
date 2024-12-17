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

nvidia-smi
lscpu
date

for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
do
	python3 onnx_testrun.py --batch $j --network mlp --cluster $1 --amount 100 --old_cluster 7234748 --version 1.1.0
done


date

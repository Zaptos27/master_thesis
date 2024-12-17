#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

export TVM_HOME=/eos/user/m/mimodekj/env/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

date
mkdir benchmarks
mkdir output
mkdir code

lscpu
date

#for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
#do
#	python3 benchmark.py --batch $j --network mlp_single --cluster $1 --code_generation --target opencl --amount 819200 --old_cluster 7225626
#done
#date
for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 16384 32768 65536
do
	python3 benchmark.py --batch $j --network mlp_single --cluster $1 --code_generation --target llvm --amount 50 --old_cluster 7234748 --version 1.1.0
	date
done
python3 benchmark.py --batch 8192 --network mlp_single --cluster $1 --code_generation --target llvm --amount 50 --old_cluster 7234748 --version 1.1.0 --output
cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536
do
	python3 benchmark.py --batch $j --network mlp_single --cluster $1 --code_generation --target opencl --amount 50 --old_cluster 7234748 --version 1.1.0
	date
done


cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
#date
#for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
#do
#	python3 onnx_testrun.py --batch $j --network mlp --cluster $1 --amount 819200 --old_cluster 7225626
#done

#cp -r onnx_benchmark/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

date

#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

export TVM_HOME=/eos/user/m/mimodekj/env/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

export TF_ENABLE_ONEDNN_OPTS=0

mkdir benchmarks
mkdir output
mkdir code

ls /eos/user/m/mimodekj/tensorflow_datasets

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

nvidia-smi
lscpu

for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.0.0 --old_cluster 7364356
	done
	python3 benchmark.py --batch 16384 --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.0.0 --output --old_cluster 7364356

	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target llvm --amount 200 --version 0.0.0 --old_cluster 7364356
	done
done

cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.1.0 --old_cluster 7364357
	done
	python3 benchmark.py --batch 16384 --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.1.0 --output --old_cluster 7364357

	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target llvm --amount 200 --version 0.1.0 --old_cluster 7364357
	done
done

cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.0.1 --old_cluster 7364359
	done
	python3 benchmark.py --batch 16384 --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.0.1 --output --old_cluster 7364359

	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target llvm --amount 200 --version 0.0.1 --old_cluster 7364359
	done
done

cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.1.1 --old_cluster 7364358
	done
	python3 benchmark.py --batch 16384 --network $i --cluster $1 --code_generation --target cuda --amount 200 --version 0.1.1 --output --old_cluster 7364358

	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --code_generation --target llvm --amount 200 --version 0.1.1 --old_cluster 7364358
	done
done

cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/


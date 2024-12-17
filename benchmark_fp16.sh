#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda2

export TVM_HOME=/eos/user/m/mimodekj/env/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

export TF_ENABLE_ONEDNN_OPTS=0

mkdir benchmarks
mkdir output
mkdir code

ls /eos/user/m/mimodekj/tensorflow_datasets

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

clusters=(7473664 7473665 7473667 7473666)

nvidia-smi
lscpu
if [ $2 == 0 ]
then
for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise mlp_big mlp_piecewise2
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target cuda --amount 100 --version 0.0.0 --old_cluster ${clusters[$2]} --fp16
	done

	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target llvm --amount 100 --version 0.0.0 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target opencl --amount 100 --version 0.0.0 --old_cluster ${clusters[$2]} --fp16
	done
done
fi
if [ $2 == 1 ]
then
for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise mlp_big mlp_piecewise2
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target cuda --amount 100 --version 0.1.0 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target llvm --amount 100 --version 0.1.0 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target opencl --amount 100 --version 0.1.0 --old_cluster ${clusters[$2]} --fp16
	done
done
fi
if [ $2 == 2 ]
then
for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise mlp_big mlp_piecewise2
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target cuda --amount 100 --version 0.0.1 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target llvm --amount 100 --version 0.0.1 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target opencl --amount 100 --version 0.0.1 --old_cluster ${clusters[$2]} --fp16
	done
done
fi
if [ $2 == 3 ]
then
for i in mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_repeat separable_conv1d separable_conv2d mlp_piecewise mlp_big mlp_piecewise2
do
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target cuda --amount 100 --version 0.1.1 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target llvm --amount 100 --version 0.1.1 --old_cluster ${clusters[$2]} --fp16
	done
	for j in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
	do
		python3 benchmark.py --batch $j --network $i --cluster $1 --target opencl --amount 100 --version 0.1.1 --old_cluster ${clusters[$2]} --fp16
	done
done
fi


cp -r benchmarks/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r output/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r code/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
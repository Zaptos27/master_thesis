#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch

eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  

networks=(mlp mlp_wide mlp_deep conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_piecewise mlp_repeat separable_conv1d separable_conv2d)

if [ $2 -lt 3 ]
then
	python3 training/training.py --epoch 1000 --batch 4096 --modelName ${networks[$2]} --modelType MLP --cluster $1 --version 0.0.0
else
	python3 training/training.py --epoch 1000 --batch 4096 --modelName ${networks[$2]} --modelType MODELS --cluster $1 --version 0.0.0 
fi


cp -r checkpoints/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r models/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r loss_info/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

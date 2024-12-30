#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch

eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda2

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  

networks=(mlp mlp_wide mlp_deep mlp_big conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_piecewise mlp_repeat separable_conv1d separable_conv2d mlp_piecewise2 mlp_small)

if [ $2 -lt 4 ]
then
	python3 training/training.py --epoch 2000 --batch 1024 --modelName ${networks[$2]} --modelType MLP --cluster $1 --version 0.1.1
else
	python3 training/training.py --epoch 2000 --batch 1024 --modelName ${networks[$2]} --modelType MODELS --cluster $1 --version 0.1.1 
fi


cp -r checkpoints/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r models/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r loss_info/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

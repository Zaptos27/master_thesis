#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch

eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  

python3 training/training.py --windowSize 3 --epoch 500 --batch 4096 --modelName mlp --modelType MLP  --cluster 11 --version 0.0.0


cp -r checkpoints/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r models/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/
cp -r loss_info/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion/

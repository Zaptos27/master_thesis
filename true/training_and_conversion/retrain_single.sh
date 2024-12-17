#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch

eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  

nvidia-smi
lscpu
date
python3 training/training.py --windowSize 4 --epoch 120 --batch 4096 --modelName mlp --modelType MLP --steps 20000 --cluster $1 --version 1.1.0 --retrain --old_cluster 7225626
date
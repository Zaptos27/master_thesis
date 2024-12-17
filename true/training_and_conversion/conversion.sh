#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate /eos/user/m/mimodekj/env/conda

python3 convert2onnx.py --windowsize 3 --total_lenght 20 --old_cluster 
cp -r models/ /afs/cern.ch/user/m/mimodekj/public/master_thesis/training_and_conversion

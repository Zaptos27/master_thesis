#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate

pip install tensorflow[and-cuda]==2.15.* apache_beam tensorflow_datasets
python3 training/training.py --windowSize 3 --epoch 100 --batch 2048 --modelName mlp --modelType MLP

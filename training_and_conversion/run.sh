#!/bin/bash

# This script is used to run multiple jobs, for training purposes in sequence

for type in MLP MODELS
do
    echo "${type}"
    if [[ "${type}" == "MLP" ]]; then
        for network in mlp mlp_wide mlp_deep
        do
            python training/training.py --windowSize 3 --epoch 1 --batch 2048 --modelName $network --modelType $type
        done
    fi
    if [[ "${type}" == "MODELS" ]]; then
        for network in mlp_deep_wide mlp_conv conv1d conv2d depthwise_conv1d separable_conv2d separable_conv1d mlp_piecewise convlstm1d lstm mlp_repeat rnn gru stacked_rnn stacked_gru conv1d_rnn
        do
            python training/training.py --windowSize 3 --epoch 1 --batch 2048 --modelName $network --modelType $type
        done
    fi

done
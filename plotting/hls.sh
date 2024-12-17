#!/bin/bash

# Run the HLS script for the given HLS file and the given output directory
# Usage: ./hls.sh <VERSION> <OLD_CLUSTER>
# Example: ./hls.sh 1.0.0 1

# Set the version and the old cluster
VERSION=$1
OLD_CLUSTER=$2

# Run the python script for each model model_names = ["mlp", "mlp_wide", "mlp_deep", "conv1d", "lstm", "mlp_conv", "mlp_deep_wide", "mlp_repeat", "separable_conv1d", "separable_conv2d", "mlp_piecewise", "conv2d" , "depthwise_conv1d", "mlp_piecewise2", "mlp_big"]
for model in mlp_small conv1d conv2d mlp_conv separable_conv1d separable_conv2d depthwise_conv1d mlp_piecewise mlp_piecewise2 mlp mlp_wide mlp_deep lstm mlp_deep_wide mlp_big mlp_repeat 
do
    python3 run_hls4ml.py --version $VERSION --model_name $model --old_cluster $OLD_CLUSTER
done
#!/bin/bash
# This script is used to run multiple jobs, for benchmarking purposes in sequence
#conda activate tvm-build
# Missing tvm support conv1d_rnn, convlstm1d, gru, rnn, stacked_gru, stacked_rnn
for network in conv1d conv2d depthwise_conv1d lstm mlp_conv mlp_deep_wide mlp_deep mlp_piecewise mlp_repeat mlp_wide mlp separable_conv1d separable_conv2d conv1d_rnn convlstm1d gru rnn stacked_gru stacked_rnn
do
    for target in llvm opencl cuda
    do
        for batch_size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
        do
            python benchmark.py --network=$network --target=$target --batch_size=$batch_size
        done
    done
done
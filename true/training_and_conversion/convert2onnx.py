import tensorflow as tf
import tf2onnx
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--windowsize", type=int, default=3, help="Window size used in the input")
parser.add_argument("--total_lenght", type=int, default=20, help="Amount of stacked layers for the big conversion")
parser.add_argument("--old_cluster", type=str, default="")

args = parser.parse_args()


window_size = args.windowsize
total_lenght = args.total_lenght

networks = ["mlp"]# ["mlp", "mlp_wide", "mlp_deep", "mlp_deep_wide", "mlp_conv", "conv1d", "conv2d", "depthwise_conv1d", "separable_conv1d", "mlp_piecewise", "lstm", "mlp_repeat", "separable_conv2d"]
for network in networks:
    init_model = tf.keras.models.load_model(f"models/tensorflow/{network}{args.old_cluster}/")
    checkpoint_filepath = 'checkpoints/' + network + args.old_cluster+'/'+network + args.old_cluster 
    init_model.load_weights(checkpoint_filepath)
    #nn_input = tf.keras.layers.Input(shape=(window_size, 3), name="input_1")
    # Make a tensor with the N x 3 shape with input data as the first 3 rows
    # Run the model on the rolling window of 3 rows
    #_x = init_model(nn_input)
    #nn_output = tf.concat((nn_input, tf.expand_dims(_x,axis=1)), axis=1)
    #for i in range(1,total_lenght-window_size):
    #    _x = init_model(tf.slice(nn_output, begin = [0, i, 0], size = [-1, window_size, -1]))
    #    nn_output = tf.concat((nn_output, tf.expand_dims(_x,axis=1)), axis=1)
    #model = tf.keras.models.Model(inputs=nn_input, outputs=nn_output)
    #tf2onnx.convert.from_keras(model, output_path=f"models/onnx/{network}{args.old_cluster}.onnx")
    tf2onnx.convert.from_keras(init_model, output_path=f"models/onnx/{network}_single{args.old_cluster}.onnx")

import hls4ml
import numpy as np
import tensorflow as tf
import pickle as pkl
import argparse
import os
os.environ['PATH'] += os.pathsep + "/mnt/HDD/Xilinx2/Vivado/2020.1/bin:$PATH"

# Set tf to use CPU
tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser(description='Convert models to hls4ml')
parser.add_argument('--amount', type=int, default=1000, help='Amount of data to use')
parser.add_argument('--old_cluster', type=str, default="7473664", help='Old cluster to use')
parser.add_argument('--version', type=str, default="0.0.0", help='Version of the dataset')
parser.add_argument('--model_name', type=str, default="mlp", help='Model to use')
args = parser.parse_args()

old_cluster = args.old_cluster
version = args.version
model_name = args.model_name

if version[2] == "1":
    windowsize = 4
else:
    windowsize = 3


data_dir = "../tensorflow_datasets"
if version[0] != "0":
    raise ValueError("Only version 0 is supported")
else:
    if version[-1] == "1":
        X_test,y_test = np.load(f"../../Data/test_data_{windowsize}_backwards/test_input.npy"),np.load(f"../../Data/test_data_{windowsize}_backwards/test_target.npy")
        X_id = np.load(f"../../Data/test_data_{windowsize}_backwards/test_id.npy")
    else:
        X_test,y_test = np.load(f"../../Data/test_data_{windowsize}/test_input.npy"),np.load(f"../../Data/test_data_{windowsize}/test_target.npy")
        X_id = np.load(f"../../Data/test_data_{windowsize}/test_id.npy")

# set tensorflow to verbose 0
tf.get_logger().setLevel('ERROR')

# load keras model
print(model_name, version, old_cluster)
model = tf.keras.models.load_model(f"../../final/models/tensorflow/{model_name}_{version}_{old_cluster}", compile=False)
print(model.input_shape)
model.load_weights(f"../../final/checkpoints/{model_name}{old_cluster}/{model_name}{old_cluster}")
model.compile()
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
config['Model']['Strategy'] = 'Resource'
total_weights = 0
for layer in model.get_weights():
    total_weights += layer.size
print("Total weights: {}".format(total_weights))
if total_weights < 50_000:
    config['Model']['ReuseFactor'] = 8
elif total_weights < 100_000:
    config['Model']['ReuseFactor'] = 16
else:
    config['Model']['ReuseFactor'] = 64
if "small" in model_name:
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Latency'

hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=f'hls4ml/{model_name}_{version}_{old_cluster}_hls4ml_prj',  part='xcu250-figd2104-2L-e', io_type='io_stream')
hls_model.compile()
hls_model.build(csim=False)
if "small" in model_name:
    y = hls_model.predict(X_test.reshape((-1, *model.input_shape[1:])))
else:
    y = hls_model.predict([X_test.reshape((-1, *model.input_shape[0][1:])), X_id])
print(model_name, version, ":   " ,np.std(y[:,0]*1015 - y_test[:,0]*1015),"  ", np.std(y[:,1]*1015 - y_test[:,1]*1015),"  " ,np.std(y[:,2]*3000 - y_test[:,2]*3000))
# Save the hls model to disk and y to disk
with open(f"hls4ml/{model_name}_{version}_{old_cluster}_y.pkl", "wb") as f:
    pkl.dump(y, f)

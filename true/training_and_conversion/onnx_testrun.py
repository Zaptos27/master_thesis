import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="mlp", help="Network name")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--cluster", type=str, default="", help="For naming")
parser.add_argument("--amount", type=int, default=-1, help="Number of datapoints used")
parser.add_argument("--old_cluster", type=str, default="")
parser.add_argument("--version", type=str, default="0.0.0")

args = parser.parse_args()

import onnxruntime as ort
import torch
import tensorflow_datasets as tfds
import numpy as np
import onnx
import time

data_dir = "/eos/user/m/mimodekj/tensorflow_datasets"



if args.version[3] == "1":
    windowsize = 4
else:
    windowsize = 3
if args.version[0] == "0":
    if args.version[-1] == "1":
        data = np.load(data_dir+f"/../test_{windowsize}_backwards/input.npy")
        data_output = np.load(data_dir+f"/../test_{windowsize}_backwards/target.npy")
    else:
        data = np.load(data_dir+f"/../test_{windowsize}/input.npy")
        data_output = np.load(data_dir+f"/../test_{windowsize}/target.npy")
    # Change datatype to float32
    data = data.astype(np.float32)
    data_output = data_output.astype(np.float32)
    #data = np.array(np.split(data[:data.shape[0]//args.batch_size*args.batch_size],data.shape[0]//args.batch_size))
    #data_output = np.array(np.split(data_output[:data_output.shape[0]//args.batch_size*args.batch_size],data_output.shape[0]//args.batch_size))
else:
    data = tfds.load(f'particle_data:{args.version}', data_dir=data_dir, split='test', as_supervised=True, shuffle_files=False).batch(args.batch_size).take(args.amount)
    num_data = tfds.as_numpy(data)
    data = [d[0] for d in num_data]
    data_output = [d[1] for d in num_data]
    data = np.stack(data)
    data_output = np.stack(data_output)

print(f"models/onnx/{args.network}{args.old_cluster}.onnx")
model = onnx.load(f"models/onnx/{args.network}{args.old_cluster}.onnx")
onnx.checker.check_model(model)




providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True 
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)

print('CUDA 1')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

time.sleep(1)

sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('CUDA 2')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

time.sleep(1)

sess.set_providers(['CPUExecutionProvider'])
print('CPU 1')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

time.sleep(1)

providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True 
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model = onnx.load(f"models/onnx/{args.network}{args.old_cluster}.onnx")
onnx.checker.check_model(model)



sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)

print('CUDA 3')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

time.sleep(1)

sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('CUDA 4')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

time.sleep(1)

sess.set_providers(['CPUExecutionProvider'])
print('CPU 5')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))



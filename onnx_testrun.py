import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="mlp", help="Network name")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--cluster", type=str, default="", help="For naming")
parser.add_argument("--amount", type=int, default=-1, help="Number of datapoints used")
parser.add_argument("--old_cluster", type=str, default="")
parser.add_argument("--version", type=str, default="1.0.0")

args = parser.parse_args()

import onnxruntime as ort
import torch
import tensorflow_datasets as tfds
import numpy as np
import onnx

#data_dir = "/eos/user/m/mimodekj/tensorflow_datasets"

#data = tfds.load(f'particle_data:{args.version}', data_dir=data_dir, split='test', as_supervised=True, shuffle_files=False).batch(args.batch_size).take(args.amount)
#num_data = tfds.as_numpy(data)
#data = [d[0] for d in num_data]
#data_output = [d[1] for d in num_data]

#data = np.stack(data)
#data_output = np.stack(data_output)
data = np.load("../../Data/test_3/input.npy")
data_output = np.load("../../Data/test_3/target.npy")
# Change datatype to float32
data = data.astype(np.float32)
data_output = data_output.astype(np.float32)


print(f"models/onnx/{args.network}_single{args.old_cluster}.onnx")
model = onnx.load(f"models/onnx/{args.network}_single{args.old_cluster}.onnx")
print(model)
onnx.checker.check_model(model)


providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True 
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)
print(ort.get_device())
print("Input Name:", sess.get_inputs()[0].name)
print("Input Shape:", sess.get_inputs()[0].shape)

# Print the output name and shape
print("Output Name:", sess.get_outputs()[0].name)
print("Output Shape:", sess.get_outputs()[0].shape)

print('CUDA 1')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)

sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('CUDA 2')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)

sess.set_providers(['CPUExecutionProvider'])
print('CPU 1')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)


providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True 
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)
print(ort.get_device())

print('CUDA 3')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)

sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('CUDA 4')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)

sess.set_providers(['CPUExecutionProvider'])
print('CPU 5')
input_name = sess.get_inputs()[0].name
result = sess.run([sess.get_outputs()[0].name], {input_name: data})
print(result)

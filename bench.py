import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="mlp", help="Network name")
parser.add_argument("--target", type=str, default="opencl", help="Target device Options opencl, cuda, llvm")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--data_type", type=str, default="float32", help="Data type")
parser.add_argument("--number", type=int, default=5, help="Number of runs")
parser.add_argument("--repeat", type=int, default=10, help="Number of repeats")
parser.add_argument("--cooldown_interval_ms", type=int, default=50, help="Cooldown interval in ms")
parser.add_argument("--min_repeat_ms", type=int, default=50, help="Minimum repeat in ms")
parser.add_argument("--FPGA", action='store_true', help="FPGA")
parser.add_argument("--model_tuning", action='store_true', help="Model tuning")
parser.add_argument("--log_file", type=str, default=None, help="Log file")
parser.add_argument("--code_generation", action='store_true', help="Code generation")
parser.add_argument("--cluster", type=str, default="", help="For naming")

args = parser.parse_args()
log_file = args.log_file


if args.FPGA:
    dpu_target = 'DPUCADF8H'
    import pyxir
    import pyxir.contrib.target.DPUCADF8H

import tvm
import onnx
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
import tvm.contrib.graph_executor as runtime
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


if args.FPGA:
    from tvm.contrib.target import vitis_ai
    from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai


# Check if folder exists
if not os.path.exists("benchmarks"):
    os.makedirs("benchmarks")
if not os.path.exists("output"):
    os.makedirs("output")
if not os.path.exists("code"):
    os.makedirs("code")

data_dir = "/mnt/HDD/tensorflow_datasets"
input_name = "input_1"

data = tfds.load('particle_data', data_dir=data_dir, split='test', as_supervised=True, shuffle_files=False).batch(args.batch_size)
num_data = tfds.as_numpy(data)
data = [d[0] for d in num_data]
print(len(data))
data_output = [d[1] for d in num_data]

data = np.stack(data)
data_output = np.stack(data_output)


onnx_model = onnx.load(f"models/onnx/{args.network}.onnx")
input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input][0]
output_shapes = [[d.dim_value for d in _output.type.tensor_type.shape.dim] for _output in onnx_model.graph.output][0]

input_shapes[0] = args.batch_size
output_shapes[0] = args.batch_size

tvm_model, params = tvm.relay.frontend.from_onnx(onnx_model, {input_name: input_shapes}, dtype=args.data_type, freeze_params=True)

if args.target == "opencl":
    file_extension = "cl"
if args.target == "cuda":
    file_extension = "cu"

if args.cluster == "":
    output_file = f"{args.network}-{args.batch_size}-{args.target}"
else:
    output_file = f"{args.network}-{args.batch_size}-{args.target}-{args.cluster}"

def evaluate_model(lib, input_shapes, input_name="input_1", data_tvm=None, number=10, repeat=600, data_transfer = False, cooldown_interval_ms=10,filename="benchmark.txt", directory="benchmarks"):
    # Load set_input
    dev = tvm.device(args.target)
    module = runtime.GraphModule(lib["default"](dev))
    if data_tvm is None:
        data_tvm = (np.random.uniform(size=(5,*input_shapes))).astype(args.data_type)
        print("Generated random data")
    
    # Evaluate
    print("Evaluate inference time cost...")
    if list(data_tvm.shape) != input_shapes:
        _benchmark = []
        for f in data_tvm:
            module.set_input(input_name, f) # Set the input data
            _benchmark.append(module.benchmark(dev, number=number, repeat=repeat, end_to_end=data_transfer, cooldown_interval_ms=cooldown_interval_ms, min_repeat_ms=args.min_repeat_ms).results)
        _benchmark = np.array(_benchmark).flatten()
        print("Evaluate inference time cost done")
        # Write a function to return the mean and std of the benchmark results
        np.savetxt(filename, _benchmark, delimiter=",")
        return _benchmark    # Fix the formatting of these results
    else:
        module.set_input(input_name, data_tvm)
        _benchmark = module.benchmark(dev, number=number, repeat=repeat, end_to_end=data_transfer, cooldown_interval_ms=cooldown_interval_ms, min_repeat_ms=args.min_repeat_ms).results
        print("Evaluate inference time cost done")
        return np.array(_benchmark)


def performance(lib, input_shapes, data_tvm=None, input_name="input_1", number=1, repeat=600):
    # Random "data"
    if data_tvm is None:
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shapes)).astype(args.data_type))
    # Load set_input
    dev = tvm.device(args.target)
    module = runtime.GraphModule(lib["default"](dev))
    if list(data_tvm.shape) != input_shapes:
        module.set_input(input_name, data_tvm[0])
    else:
        module.set_input(input_name, data_tvm)
    module.run()
    if list(data_tvm.shape) != input_shapes:
        predicted_output = module.get_output(0).asnumpy()
        for i in range(1, data_tvm.shape[0]):
            module.set_input(input_name, data_tvm[i])
            module.run()
            predicted_output = np.concatenate((predicted_output, module.get_output(0).asnumpy()), axis=0)
    else:
        predicted_output = module.get_output(0).asnumpy()

    return predicted_output


if args.FPGA:
    tvm_model = partition_for_vitis_ai(tvm_model, params, dpu=dpu_target)
    export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
    build_options = {
        'dpu': dpu_target,
        'export_runtime_module': export_rt_mod_file
    }
    with tvm.transform.PassContext(opt_level=3, config={'tvm.relay.ext.vitis_ai.options': build_options}):
        lib = tvm.relay.build_module.build(tvm_model, target=args.target, params=params)
    lib.export_library("vitis_ai.so")
    lib.get_lib().imported_modules[1]
    module = runtime.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(input_name, tvm.nd.array((np.random.uniform(size=input_shapes)).astype(args.data_type)))
    module.run()
    print("Evaluate inference time cost...")
    print(module.benchmark(tvm.cpu()))
else:
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build_module.build(tvm_model, target=args.target, params=params)

    # Write unoptimized graph to file
    if args.code_generation and args.target != "llvm":
        with open("code/"+output_file+"-unoptimized."+file_extension, "w") as f:
            f.write(lib.get_lib().imported_modules[0].get_source())

    # Write unoptimized performance to file
    benchmark = evaluate_model(lib, input_shapes, data_tvm=data, input_name=input_name, number=args.number, repeat=args.repeat, cooldown_interval_ms=args.cooldown_interval_ms, filename=output_file+"-unoptimized"+".perf")
    
    
    # Print mean, median, std, min, max
    print(f"Mean: {np.mean(benchmark)}")
    print(f"Median: {np.median(benchmark)}")
    print(f"Std: {np.std(benchmark)}")
    print(f"Min: {np.amin(benchmark)}")
    print(f"Max: {np.amax(benchmark)}")


    benchmark = evaluate_model(lib, input_shapes, data_tvm=data, input_name=input_name, data_transfer=True, number=args.number, repeat=args.repeat, cooldown_interval_ms=args.cooldown_interval_ms, filename=output_file+"-unoptimized"+"-data_transfer"+".perf")
    # Print mean, median, std, min, max
    print("Data transfer:")
    print(f"Mean: {np.mean(benchmark)}")
    print(f"Median: {np.median(benchmark)}")
    print(f"Std: {np.std(benchmark)}")
    print(f"Min: {np.amin(benchmark)}")
    print(f"Max: {np.amax(benchmark)}")

    # Optimize
    output = performance(lib, input_shapes, data_tvm=data, input_name=input_name)
#    if args.plotting:
#        plot_curve(output, 0.95)
    np.save("output/"+output_file+"-unoptimized", output)
    np.save("output/"+output_file+"-true", data_output)
    
if args.model_tuning and not args.FPGA:
    if log_file is None:
        log_file = "%s.log" % (args.network+"_"+args.target+"_"+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1000,
        "early_stopping": 1000,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=args.number*10, repeat=args.repeat*5, timeout=5, min_repeat_ms=args.min_repeat_ms, cooldown_interval_ms=args.cooldown_interval_ms),
        ),
    }

    def tune_model(model, params, tuning_option, target, log_file=None):
        if log_file is None:
            log_file = tuning_option["log_filename"]
        # Create tmp log file
        tmp_log_file = log_file + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        # Create tasks
        tasks = autotvm.task.extract_from_program(
            model["main"], target=target, params=params,
        )

        # Tune tasks
        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(tsk, loss_type="reg")
            if os.path.exists(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

            
            tuner_obj.tune(
                n_trial=tuning_option["n_trial"],
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(tuning_option["n_trial"], prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_file)
        os.remove(tmp_log_file)
    
    tune_model({"main": tvm_model}, params, tuning_option, args.target)


if log_file is not None:
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build_module.build(tvm_model, target=args.target, params=params)

        # Write optimized graph to file
        if args.code_generation and args.target != "llvm":
            with open(log_file.replace(".log", "."+file_extension), "w") as f:
                f.write(lib.get_lib().imported_modules[0].get_source())

        # Write optimized performance to file
        benchmark = evaluate_model(lib, input_shapes, data_tvm=data, input_name=input_name, number=args.number, repeat=args.repeat, cooldown_interval_ms=args.cooldown_interval_ms, min_repeat_ms=args.min_repeat_ms, filename=log_file.replace(".log", ".perf"))

        # Print mean, median, std, min, max
        print(f"Mean: {np.mean(benchmark)}")
        print(f"Median: {np.median(benchmark)}")
        print(f"Std: {np.std(benchmark)}")
        print(f"Min: {np.amin(benchmark)}")
        print(f"Max: {np.amax(benchmark)}")
        
        benchmark = evaluate_model(lib, input_shapes, data_tvm=data, input_name=input_name, data_transfer=True, number=args.number, repeat=args.repeat, cooldown_interval_ms=args.cooldown_interval_ms, min_repeat_ms=args.min_repeat_ms, filename=log_file.replace(".log", "-data_transfer.perf"))
        
        # Print mean, median, std, min, max
        print("Data transfer:")
        print(f"Mean: {np.mean(benchmark)}")
        print(f"Median: {np.median(benchmark)}")
        print(f"Std: {np.std(benchmark)}")
        print(f"Min: {np.amin(benchmark)}")
        print(f"Max: {np.amax(benchmark)}")

        output = performance(lib, input_shapes, data_tvm=data, input_name=input_name)
#        if args.plotting:
#            plot_curve(output, 0.95)
        np.save(output_file, output)

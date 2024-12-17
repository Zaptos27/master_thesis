import uproot
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import argparse 
import os


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/eos/user/m/mimodekj/")
parser.add_argument("--window_size", type=int, default=3)
parser.add_argument("--pad_lenght", type=int, default=3)
parser.add_argument("--backwards", action="store_true", default=False)

args = parser.parse_args()

path = args.path

l = glob.glob(path+"raw_root/**/*.root", recursive=True)

branches = ["SP_x", "SP_y", "SP_z"]


window_size = args.window_size
pad_lenght = args.pad_lenght
backwards = args.backwards

if backwards:
    if not os.path.exists(f"{path}/train_data_{window_size}_backwards"):
        os.makedirs(f"{path}/train_data_{window_size}_backwards")
    if not os.path.exists(f"{path}/val_data_{window_size}_backwards"):
        os.makedirs(f"{path}/val_data_{window_size}_backwards")
    if not os.path.exists(f"{path}/test_data_{window_size}_backwards"):
        os.makedirs(f"{path}/test_data_{window_size}_backwards")
    if not os.path.exists(f"{path}/train_{window_size}_backwards"):
        os.makedirs(f"{path}/train_{window_size}_backwards")
    if not os.path.exists(f"{path}/val_{window_size}_backwards"):
        os.makedirs(f"{path}/val_{window_size}_backwards")
    if not os.path.exists(f"{path}/test_{window_size}_backwards"):
        os.makedirs(f"{path}/test_{window_size}_backwards")
else:
    if not os.path.exists(f"{path}/train_data_{window_size}"):
        os.makedirs(f"{path}/train_data_{window_size}")
    if not os.path.exists(f"{path}/val_data_{window_size}"):
        os.makedirs(f"{path}/val_data_{window_size}")
    if not os.path.exists(f"{path}/test_data_{window_size}"):
        os.makedirs(f"{path}/test_data_{window_size}")
    if not os.path.exists(f"{path}/train_{window_size}"):
        os.makedirs(f"{path}/train_{window_size}")
    if not os.path.exists(f"{path}/val_{window_size}"):
        os.makedirs(f"{path}/val_{window_size}")
    if not os.path.exists(f"{path}/test_{window_size}"):
        os.makedirs(f"{path}/test_{window_size}")


for j, file in enumerate(l):
    inp_train = np.empty((0,window_size,3))
    tar_train = np.empty((0,3))
    inp_val = np.empty((0,window_size,3))
    tar_val = np.empty((0,3))
    inp = np.empty((0,window_size,3))
    tar = np.empty((0,3))


    with uproot.open(file) as f:
        coordinateTree = f["SPInfo"]
        coordinate_dict = coordinateTree.arrays(branches, library='np')
    c1 = [c for c in coordinate_dict[branches[0]] if len(c) > 0]
    c2 = [c for c in coordinate_dict[branches[1]] if len(c) > 0]
    c3 = [c for c in coordinate_dict[branches[2]] if len(c) > 0]
    if args.backwards:
        c1 = c1[::-1]
        c2 = c2[::-1]
        c3 = c3[::-1]

    num_tracks = len(c1)
    for i in range(num_tracks):
        c1_arr = c1[i].reshape(-1,1) / 1015
        c2_arr = c2[i].reshape(-1,1) / 1015
        c3_arr = c3[i].reshape(-1,1) / 3500
        track = np.hstack((c1_arr, c2_arr, c3_arr))
        _, idx = np.unique(track, axis=0, return_index=True)
        track = track[np.sort(idx)]
        # Pad the track with 0s pad_lenght times
        track = np.pad(track, ((0, pad_lenght),(0,0)))
    
        # Reshape into rolling window of window_size
        track = np.lib.stride_tricks.sliding_window_view(track, (window_size,3)).reshape(-1,window_size,3)


        # Split the track into input and target and split into training, validation and test sets
        if np.array([t[-1] for t in track[1:]]).shape[0] and track[:-1].shape[0]:
            if 0.85*num_tracks > i:
                inp_train = np.vstack((inp_train,track[:-1]))
                tar_train = np.vstack((tar_train, np.array([t[-1] for t in track[1:]])))
            elif 0.9*num_tracks > i:
                inp_val = np.vstack((inp_val,track[:-1]))
                tar_val = np.vstack((tar_val, np.array([t[-1] for t in track[1:]])))
            else:
                inp = np.vstack((inp,track[:-1]))
                tar = np.vstack((tar, np.array([t[-1] for t in track[1:]])))
    if backwards:
        np.save(f"{path}/train_data_{window_size}_backwards/{j}_input.npy", inp_train)
        np.save(f"{path}/train_data_{window_size}_backwards/{j}_target.npy", tar_train)
        np.save(f"{path}/val_data_{window_size}_backwards/{j}_input.npy", inp_val)
        np.save(f"{path}/val_data_{window_size}_backwards/{j}_target.npy", tar_val)
        np.save(f"{path}/test_data_{window_size}_backwards/{j}_input.npy", inp)
        np.save(f"{path}/test_data_{window_size}_backwards/{j}_target.npy", tar)
    else:
        np.save(f"{path}/train_data_{window_size}/{j}_input.npy", inp_train)
        np.save(f"{path}/train_data_{window_size}/{j}_target.npy", tar_train)
        np.save(f"{path}/val_data_{window_size}/{j}_input.npy", inp_val)
        np.save(f"{path}/val_data_{window_size}/{j}_target.npy", tar_val)
        np.save(f"{path}/test_data_{window_size}/{j}_input.npy", inp)
        np.save(f"{path}/test_data_{window_size}/{j}_target.npy", tar)


if backwards:
    l_input = glob.glob(f"{path}/train_data_{window_size}_backwards/*_input.npy", recursive=True)
    l_target = glob.glob(f"{path}/train_data_{window_size}_backwards/*_target.npy", recursive=True)
    l_input_val = glob.glob(f"{path}/val_data_{window_size}_backwards/*_input.npy", recursive=True)
    l_target_val = glob.glob(f"{path}/val_data_{window_size}_backwards/*_target.npy", recursive=True)
else:
    l_input = glob.glob(f"{path}/train_data_{window_size}/*_input.npy", recursive=True)
    l_target = glob.glob(f"{path}/train_data_{window_size}/*_target.npy", recursive=True)
    l_input_val = glob.glob(f"{path}/val_data_{window_size}/*_input.npy", recursive=True)
    l_target_val = glob.glob(f"{path}/val_data_{window_size}/*_target.npy", recursive=True)

l_input.sort()
l_target.sort()
l_input_val.sort()
l_target_val.sort()


train_data = np.empty((0,window_size,3))
train_target = np.empty((0,3))
file_index = 0
for i in range(len(l_input)):
    train_data = np.vstack((train_data, np.load(l_input[i])))
    train_target = np.vstack((train_target, np.load(l_target[i])))
    if i % 6 == 0 and i != 0:
        index = np.arange(train_data.shape[0])
        np.random.shuffle(index)
        np.save(f"{path}/train_index_{window_size}_{i}.npy", index)
        train_data = train_data[index]
        train_target = train_target[index]
        split = int(np.ceil(train_data.nbytes/1.5e9))
        train_data = np.array_split(train_data, split)
        train_target = np.array_split(train_target, split)
        for j, f in enumerate(train_data):
            print(j+file_index)
            print(train_data.shape)
            print(train_target.shape)
            np.save(f"{path}/train_{window_size}/{j+file_index}_input.npy", f)
            np.save(f"{path}/train_{window_size}/{j+file_index}_target.npy", train_target[i])
        file_index += j
        train_data = np.empty((0,window_size,3))
        train_target = np.empty((0,3))

del train_data, train_target

val_data = np.empty((0,window_size,3))
val_target = np.empty((0,3))

for i in range(len(l_input_val)):
    val_data = np.vstack((val_data, np.load(l_input_val[i])))
    val_target = np.vstack((val_target, np.load(l_target_val[i])))
index = np.arange(val_data.shape[0])
np.random.shuffle(index)
np.save(f"{path}/val_index_{window_size}.npy", index)
val_data = val_data[index]
val_target = val_target[index]
np.save(f"{path}/val_{window_size}/input.npy", val_data)
np.save(f"{path}/val_{window_size}/target.npy", val_target)

del val_data, val_target

if backwards:
    l_test_input = glob.glob(f"{path}/test_{window_size}_backwards/*_input.npy", recursive=True)
    l_test_target = glob.glob(f"{path}/test_{window_size}_backwards/*_target.npy", recursive=True)
else:
    l_test_input = glob.glob(f"{path}/test_{window_size}/*_input.npy", recursive=True)
    l_test_target = glob.glob(f"{path}/test_{window_size}/*_target.npy", recursive=True)

l_test_input.sort()
l_test_target.sort()

test_data = np.empty((0,window_size,3))
test_target = np.empty((0,3))
for i in range(len(l_test_input)):
    test_data = np.vstack((test_data, np.load(l_test_input[i])))
    test_target = np.vstack((test_target, np.load(l_test_target[i])))
print(test_data.shape)
print(test_target.shape)
np.save(f"{path}/test_{window_size}/input.npy", test_data)
np.save(f"{path}/test_{window_size}/target.npy", test_target)

del test_data, test_target

print("Data preparation done!")


if backwards:
    os.system(f"rm -r {path}/train_data_{window_size}_backwards")
    os.system(f"rm -r {path}/val_data_{window_size}_backwards")
    os.system(f"rm -r {path}/test_data_{window_size}_backwards")
else:
    os.system(f"rm -r {path}/train_data_{window_size}")
    os.system(f"rm -r {path}/val_data_{window_size}")
    os.system(f"rm -r {path}/test_data_{window_size}")













    
    


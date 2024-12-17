import uproot
import numpy as np
import glob
import argparse 
import os
import tensorflow as tf
import tqdm
tf.config.set_visible_devices([], 'GPU')



parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/eos/user/m/mimodekj/")
parser.add_argument("--window_size", type=int, default=3)
parser.add_argument("--backwards", action="store_true", default=False)

args = parser.parse_args()

path = args.path

l = glob.glob(path+"/**/*.root", recursive=True)

branches = ["SP_x", "SP_y", "SP_z", "SP_recoTrackIndex", "SP_r"]


window_size = args.window_size
backwards = args.backwards

if backwards:
    if not os.path.exists(f"{path}/data_{window_size}_backwards"):
        os.makedirs(f"{path}/data_{window_size}_backwards")
    if not os.path.exists(f"{path}/train_data_{window_size}_backwards"):
        os.makedirs(f"{path}/train_data_{window_size}_backwards")
    if not os.path.exists(f"{path}/val_data_{window_size}_backwards"):
        os.makedirs(f"{path}/val_data_{window_size}_backwards")
    if not os.path.exists(f"{path}/test_data_{window_size}_backwards"):
        os.makedirs(f"{path}/test_data_{window_size}_backwards")
    #if not os.path.exists(f"{path}/train_{window_size}_backwards"):
    #    os.makedirs(f"{path}/train_{window_size}_backwards")
    #if not os.path.exists(f"{path}/val_{window_size}_backwards"):
    #    os.makedirs(f"{path}/val_{window_size}_backwards")
    #if not os.path.exists(f"{path}/test_{window_size}_backwards"):
    #    os.makedirs(f"{path}/test_{window_size}_backwards")
else:
    if not os.path.exists(f"{path}/data_{window_size}"):
        os.makedirs(f"{path}/data_{window_size}")
    if not os.path.exists(f"{path}/train_data_{window_size}"):
        os.makedirs(f"{path}/train_data_{window_size}")
    if not os.path.exists(f"{path}/val_data_{window_size}"):
        os.makedirs(f"{path}/val_data_{window_size}")
    if not os.path.exists(f"{path}/test_data_{window_size}"):
        os.makedirs(f"{path}/test_data_{window_size}")
    #if not os.path.exists(f"{path}/train_{window_size}"):
    #    os.makedirs(f"{path}/train_{window_size}")
    #if not os.path.exists(f"{path}/val_{window_size}"):
    #    os.makedirs(f"{path}/val_{window_size}")
    #if not os.path.exists(f"{path}/test_{window_size}"):
    #    os.makedirs(f"{path}/test_{window_size}")

inp = np.empty((0,window_size,3))
tar = np.empty((0,3))
id = np.empty((0,216))
file_index = 0
#file_index2 = 0
for j, file in tqdm.tqdm(enumerate(l)):
    inp = np.empty((0,window_size,3))
    tar = np.empty((0,3))
    id = np.empty((0,216))


    inp_train = np.empty((0,window_size,3))
    tar_train = np.empty((0,3))
    inp_val = np.empty((0,window_size,3))
    tar_val = np.empty((0,3))
    inp = np.empty((0,window_size,3))
    tar = np.empty((0,3))
    id_train = np.empty((0,216))
    id_val = np.empty((0,216))
    id_test = np.empty((0,216))

    with uproot.open(file) as f:
        coordinateTree = f["SPInfo"]
        coordinate_dict = coordinateTree.arrays(branches, library='np')
        truthTree = f["RecoTracks"]
        clusterIdx = truthTree.arrays("track_SPIndex", library='np')["track_SPIndex"]
        trackID = coordinateTree.arrays("SP_fineID", library='np')["SP_fineID"]
        volumeID = coordinateTree.arrays("SP_volumeID", library='np')["SP_volumeID"]
        trackPt = truthTree.arrays("track_Pt", library='np')["track_Pt"]
        trackPixHoles = truthTree.arrays("track_nPixHoles", library='np')["track_nPixHoles"]
        trackSCTHoles = truthTree.arrays("track_nSCTHoles", library='np')["track_nSCTHoles"]
        hitcluster_idx = truthTree.arrays("track_clusterIndex", library='np')["track_clusterIndex"]
        detectorID = f["HitInfo"].arrays("hit_fineID", library='np')["hit_fineID"]
        
    num_tracks = len(clusterIdx)
    for event in range(len(clusterIdx)):
        for trackIdx in range(len(clusterIdx[event])):
            if trackPt[event][trackIdx] < 1000:
                continue
            if trackPixHoles[event][trackIdx] > 0: continue
            if trackSCTHoles[event][trackIdx] > 0: continue

            indices = np.array(clusterIdx[event][trackIdx], dtype=np.int32) - 1
            indicesHit = np.array(hitcluster_idx[event][trackIdx], dtype=np.int32) - 1
            IDs = trackID[event][indices]
            hit_IDs = detectorID[event][indicesHit]
            unique_IDs = np.unique(IDs, return_index=True)[1]
            unique_hit_IDs = np.unique(hit_IDs, return_index=True)[1]
            indices = indices[sorted(unique_IDs)]

            IDs = trackID[event][indices]
            if len(unique_IDs) != len(unique_hit_IDs):
                continue
            
            
            if len(indices) <= window_size:
                continue
            
            
            c1_arr = coordinate_dict[branches[0]][event][indices]
            c2_arr = coordinate_dict[branches[1]][event][indices]
            c3_arr = coordinate_dict[branches[2]][event][indices]

            r = c1_arr**2 + c2_arr**2 + c3_arr**2
            sorted_indices = np.argsort(r)
            c1_arr = c1_arr[sorted_indices].astype(np.float64)
            c2_arr = c2_arr[sorted_indices].astype(np.float64)
            c3_arr = c3_arr[sorted_indices].astype(np.float64)

            c1_arr = c1_arr.reshape(-1,1) / 1015
            c2_arr = c2_arr.reshape(-1,1) / 1015
            c3_arr = c3_arr.reshape(-1,1) / 3000

            dID = tf.one_hot(IDs[sorted_indices], 216, dtype=tf.float64).numpy()

            track = np.hstack((c1_arr, c2_arr, c3_arr), dtype=np.float32)
            if backwards:
                track = np.flip(track, axis=0)
                dID = np.flip(dID, axis=0)


            # Reshape into rolling window of window_size
            track = np.lib.stride_tricks.sliding_window_view(track, (window_size, 3)).reshape(-1, window_size, 3)

            inp = np.vstack((inp,track[:-1].copy()))
            id = np.vstack((id, dID[window_size:].copy()))
            tar = np.vstack((tar, np.array([t[-1] for t in track[1:].copy()])))

        if inp.shape[0] > 1e4:
            if backwards:
                np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_input.npy", inp)
                np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_target.npy", tar)
                np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_id.npy", id)
            else:
                np.save(f"{path}/data_{window_size}/data_{j+file_index}_input.npy", inp)
                np.save(f"{path}/data_{window_size}/data_{j+file_index}_target.npy", tar)
                np.save(f"{path}/data_{window_size}/data_{j+file_index}_id.npy", id)
            
            inp = np.empty((0,window_size,3))
            tar = np.empty((0,3))
            id = np.empty((0,216))
            #print(j+file_index)
            file_index += 1
    
    if inp.shape[0] > 0:
        if backwards:
            np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_input.npy", inp)
            np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_target.npy", tar)
            np.save(f"{path}/data_{window_size}_backwards/data_{j+file_index}_id.npy", id)
        else:
            np.save(f"{path}/data_{window_size}/data_{j+file_index}_input.npy", inp)
            np.save(f"{path}/data_{window_size}/data_{j+file_index}_target.npy", tar)
            np.save(f"{path}/data_{window_size}/data_{j+file_index}_id.npy", id)
        #print(j+file_index)
    else:
        file_index -= 1

    
    #print(inp.shape)
    #print(tar.shape)
    #print(id.shape)

    #print(inp.shape)
    #print(tar.shape)
    #print(id.shape)
    #if z == 10:
    #    exit()
    #else:
    #    z+=1

            # Split the track into input and target and split into training, validation and test sets
            #if np.array([t[-1] for t in track[1:]]).shape[0] and track[:-1].shape[0]:
            #    if 0.85*num_tracks > event:
            #        inp_train = np.vstack((inp_train,track[:-1]))
            #        id_train = np.vstack((id_train, dID[window_size:]))
            #        tar_train = np.vstack((tar_train, np.array([t[-1] for t in track[1:]])))
            #    elif 0.9*num_tracks > event:
            #        inp_val = np.vstack((inp_val,track[:-1]))
            #        id_val = np.vstack((id_val, dID[window_size:]))
            #        tar_val = np.vstack((tar_val, np.array([t[-1] for t in track[1:]])))
            #    else:
            #        inp = np.vstack((inp,track[:-1]))
            #        id_test = np.vstack((id_test, dID[window_size:]))
            #        tar = np.vstack((tar, np.array([t[-1] for t in track[1:]])))
        

    #if backwards:
    #    np.save(f"{path}/train_data_{window_size}_backwards/{j}_input.npy", inp_train)
    #    np.save(f"{path}/train_data_{window_size}_backwards/{j}_target.npy", tar_train)
    #    np.save(f"{path}/val_data_{window_size}_backwards/{j}_input.npy", inp_val)
    #    np.save(f"{path}/val_data_{window_size}_backwards/{j}_target.npy", tar_val)
    #    np.save(f"{path}/test_data_{window_size}_backwards/{j}_input.npy", inp)
    #    np.save(f"{path}/test_data_{window_size}_backwards/{j}_target.npy", tar)
    #    np.save(f"{path}/train_data_{window_size}_backwards/{j}_id.npy", id_train)
    #    np.save(f"{path}/val_data_{window_size}_backwards/{j}_id.npy", id_val)
    #    np.save(f"{path}/test_data_{window_size}_backwards/{j}_id.npy", id_test)
    #else:
    #    np.save(f"{path}/train_data_{window_size}/{j}_input.npy", inp_train)
    #    np.save(f"{path}/train_data_{window_size}/{j}_target.npy", tar_train)
    #    np.save(f"{path}/val_data_{window_size}/{j}_input.npy", inp_val)
    #    np.save(f"{path}/val_data_{window_size}/{j}_target.npy", tar_val)
    #    np.save(f"{path}/test_data_{window_size}/{j}_input.npy", inp)
    #    np.save(f"{path}/test_data_{window_size}/{j}_target.npy", tar)
    #    np.save(f"{path}/train_data_{window_size}/{j}_id.npy", id_train)
    #    np.save(f"{path}/val_data_{window_size}/{j}_id.npy", id_val)
    #    np.save(f"{path}/test_data_{window_size}/{j}_id.npy", id_test)



if backwards:
    l = glob.glob(f"{path}/data_{window_size}_backwards/*_input.npy", recursive=True)
    l_id = glob.glob(f"{path}/data_{window_size}_backwards/*_id.npy", recursive=True)
    l_target = glob.glob(f"{path}/data_{window_size}_backwards/*_target.npy", recursive=True)
else:
    l = glob.glob(f"{path}/data_{window_size}/*_input.npy", recursive=True)
    l_id = glob.glob(f"{path}/data_{window_size}/*_id.npy", recursive=True)
    l_target = glob.glob(f"{path}/data_{window_size}/*_target.npy", recursive=True)

l.sort()
l_id.sort()
l_target.sort()
inp = np.empty((0,window_size,3))
tar = np.empty((0,3))
id = np.empty((0,216))

for i in range(len(l)):
    inp = np.vstack((inp, np.load(l[i])))
    tar = np.vstack((tar, np.load(l_target[i])))
    id = np.vstack((id, np.load(l_id[i])))

index = np.arange(inp.shape[0])
np.random.shuffle(index)
np.save(f"{path}/data_index.npy", index)
inp = inp[index]
tar = tar[index]
id = id[index]
# Make a train test val split of 80 10 10
lenght = inp.shape[0]
train_lenght = int(0.8*lenght)
val_lenght = int(0.1*lenght)
inp_train = inp[:train_lenght]
tar_train = tar[:train_lenght]
id_train = id[:train_lenght]
inp_val = inp[train_lenght:train_lenght+val_lenght]
tar_val = tar[train_lenght:train_lenght+val_lenght]
id_val = id[train_lenght:train_lenght+val_lenght]
inp_test = inp[train_lenght+val_lenght:]
tar_test = tar[train_lenght+val_lenght:]
id_test = id[train_lenght+val_lenght:]

if backwards:
    np.save(f"{path}/train_data_{window_size}_backwards/train_input.npy", inp_train)
    np.save(f"{path}/train_data_{window_size}_backwards/train_target.npy", tar_train)
    np.save(f"{path}/train_data_{window_size}_backwards/train_id.npy", id_train)
    np.save(f"{path}/val_data_{window_size}_backwards/val_input.npy", inp_val)
    np.save(f"{path}/val_data_{window_size}_backwards/val_target.npy", tar_val)
    np.save(f"{path}/val_data_{window_size}_backwards/val_id.npy", id_val)
    np.save(f"{path}/test_data_{window_size}_backwards/test_input.npy", inp_test)
    np.save(f"{path}/test_data_{window_size}_backwards/test_target.npy", tar_test)
    np.save(f"{path}/test_data_{window_size}_backwards/test_id.npy", id_test)
else:
    np.save(f"{path}/train_data_{window_size}/train_input.npy", inp_train)
    np.save(f"{path}/train_data_{window_size}/train_target.npy", tar_train)
    np.save(f"{path}/train_data_{window_size}/train_id.npy", id_train)
    np.save(f"{path}/val_data_{window_size}/val_input.npy", inp_val)
    np.save(f"{path}/val_data_{window_size}/val_target.npy", tar_val)
    np.save(f"{path}/val_data_{window_size}/val_id.npy", id_val)
    np.save(f"{path}/test_data_{window_size}/test_input.npy", inp_test)
    np.save(f"{path}/test_data_{window_size}/test_target.npy", tar_test)
    np.save(f"{path}/test_data_{window_size}/test_id.npy", id_test)



exit()
lenght = inp_train.shape[0]

index = np.arange(lenght)
np.random.shuffle(index)
print(inp.shape)
np.save(f"{path}/data_inp.npy", inp)
print(tar.shape)
np.save(f"{path}/data_tar.npy", tar)
print(id.shape)
np.save(f"{path}/data_id.npy", id)
print(index.shape)
np.save(f"{path}/data_index.npy", index)

if backwards:
    l_input = glob.glob(f"{path}/train_data_{window_size}_backwards/*_input.npy", recursive=True)
    l_target = glob.glob(f"{path}/train_data_{window_size}_backwards/*_target.npy", recursive=True)
    l_id = glob.glob(f"{path}/train_data_{window_size}_backwards/*_id.npy", recursive=True)
    l_input_val = glob.glob(f"{path}/val_data_{window_size}_backwards/*_input.npy", recursive=True)
    l_target_val = glob.glob(f"{path}/val_data_{window_size}_backwards/*_target.npy", recursive=True)
    l_id_val = glob.glob(f"{path}/val_data_{window_size}_backwards/*_id.npy", recursive=True)
else:
    l_input = glob.glob(f"{path}/train_data_{window_size}/*_input.npy", recursive=True)
    l_target = glob.glob(f"{path}/train_data_{window_size}/*_target.npy", recursive=True)
    l_id = glob.glob(f"{path}/train_data_{window_size}/*_id.npy", recursive=True)
    l_input_val = glob.glob(f"{path}/val_data_{window_size}/*_input.npy", recursive=True)
    l_target_val = glob.glob(f"{path}/val_data_{window_size}/*_target.npy", recursive=True)
    l_id_val = glob.glob(f"{path}/val_data_{window_size}/*_id.npy", recursive=True)

l_input.sort()
l_target.sort()
l_input_val.sort()
l_target_val.sort()

idx = np.arange(len(l_input))
np.random.shuffle(idx)
l_input = np.array(l_input)[idx]
l_target = np.array(l_target)[idx]



train_data = np.empty((0,window_size,3))
train_target = np.empty((0,3))
train_id = np.empty((0,216))
file_index = 0
for i in range(len(l_input)):
    train_data = np.vstack((train_data, np.load(l_input[i])))
    train_target = np.vstack((train_target, np.load(l_target[i])))
    train_id = np.vstack((train_id, np.load(l_id[i])))
    if i % 60 == 0 and i != 0:
        index = np.arange(train_data.shape[0])
        np.random.shuffle(index)
        np.save(f"{path}/train_index_{window_size}_{i}.npy", index)

        train_data = train_data[index]
        train_target = train_target[index]
        train_id = train_id[index]

        split = int(np.ceil(train_data.nbytes/1.5e9))

        train_data = np.array_split(train_data, split)
        train_target = np.array_split(train_target, split)
        train_id = np.array_split(train_id, split)

        for j, f in enumerate(train_data):
            print(j+file_index)
            np.save(f"{path}/train_{window_size}/{j+file_index}_input.npy", f)
            np.save(f"{path}/train_{window_size}/{j+file_index}_target.npy", train_target[j])
            np.save(f"{path}/train_{window_size}/{j+file_index}_id.npy", train_id[j])
        file_index += j+1
        train_data = np.empty((0,window_size,3))
        train_target = np.empty((0,3))
        train_id = np.empty((0,216))
if train_data.shape[0]:
    np.save(f"{path}/train_{window_size}/{file_index+1}_input.npy", train_data)
    np.save(f"{path}/train_{window_size}/{file_index+1}_target.npy", train_target)
    np.save(f"{path}/train_{window_size}/{file_index+1}_id.npy", train_id)

print(train_data.shape)
print(train_target.shape)
print(train_id.shape)
del train_data, train_target

val_data = np.empty((0,window_size,3))
val_target = np.empty((0,3))
val_id = np.empty((0,216))

for i in range(len(l_input_val)):
    val_data = np.vstack((val_data, np.load(l_input_val[i])))
    val_target = np.vstack((val_target, np.load(l_target_val[i])))
    val_id = np.vstack((val_id, np.load(l_id_val[i])))
index = np.arange(val_data.shape[0])
np.random.shuffle(index)
np.save(f"{path}/val_index_{window_size}.npy", index)
val_data = val_data[index]
val_target = val_target[index]
np.save(f"{path}/val_{window_size}/input.npy", val_data)
np.save(f"{path}/val_{window_size}/target.npy", val_target)
np.save(f"{path}/val_{window_size}/id.npy", val_id)
print(val_data.shape)
print(val_target.shape)
print(val_id.shape)
del val_data, val_target

if backwards:
    l_test_input = glob.glob(f"{path}/test_data_{window_size}_backwards/*_input.npy", recursive=True)
    l_test_target = glob.glob(f"{path}/test_data_{window_size}_backwards/*_target.npy", recursive=True)
    l_test_id = glob.glob(f"{path}/test_data_{window_size}_backwards/*_id.npy", recursive=True)
else:
    l_test_input = glob.glob(f"{path}/test_data_{window_size}/*_input.npy", recursive=True)
    l_test_target = glob.glob(f"{path}/test_data_{window_size}/*_target.npy", recursive=True)
    l_test_id = glob.glob(f"{path}/test_data_{window_size}/*_id.npy", recursive=True)

l_test_input.sort()
l_test_target.sort()

test_data = np.empty((0,window_size,3))
test_target = np.empty((0,3))
test_id = np.empty((0,216))
for i in range(len(l_test_input)):
    test_data = np.vstack((test_data, np.load(l_test_input[i])))
    test_target = np.vstack((test_target, np.load(l_test_target[i])))
    test_id = np.vstack((test_id, np.load(l_test_id[i])))
print(test_data.shape)
print(test_target.shape)
print(test_id.shape)
np.save(f"{path}/test_{window_size}/input.npy", test_data)
np.save(f"{path}/test_{window_size}/target.npy", test_target)
np.save(f"{path}/test_{window_size}/id.npy", test_id)

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













    
    


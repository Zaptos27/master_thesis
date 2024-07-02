import numpy as np
number_of_files = 93
random = np.arange(number_of_files)
np.random.shuffle(random)
size = 10
N = int(np.ceil((number_of_files-1)/size))
for j in range(N):
    print("J: ",j+1, " of ", N)
    arr = np.empty((0,3,3))
    for i in random[(j)*size:(j+1)*size]:
        if i>number_of_files-1: continue
        _x = np.load(f"/mnt/HDD/data/{i}_input.npy")
        arr = np.append(arr, _x.copy(), axis=0)
    del _x
    index = np.arange(arr.shape[0])
    np.random.shuffle(index)
    np.save(f"/mnt/HDD/index/{j}_index.npy", index)
    arr = arr[index]
    split = int(np.ceil(arr.nbytes/1.5e9))
    arr = np.array_split(arr, split)
    for i, f in enumerate(arr):
        np.save(f"/mnt/HDD/train_data/{j}{i}_input.npy", f[:int(len(f)*0.75)])
        np.save(f"/mnt/HDD/test_data/{j}{i}_input.npy", f[int(len(f)*0.75):int(len(f)*0.9)])
        np.save(f"/mnt/HDD/val_data/{j}{i}_input.npy", f[int(len(f)*0.9):])
    arr = np.empty((0,3))
    for i in random[(j)*size:(j+1)*size]:
        if i>number_of_files-1: continue
        _x = np.load(f"/mnt/HDD/data/{i}_target.npy")
        arr = np.append(arr, _x.copy(), axis=0)
    arr = arr[index]
    arr = np.array_split(arr, split)
    for i, f in enumerate(arr):
        np.save(f"/mnt/HDD/train_data/{j}{i}_target.npy", f[:int(len(f)*0.75)])
        np.save(f"/mnt/HDD/test_data/{j}{i}_target.npy", f[int(len(f)*0.75):int(len(f)*0.9)])
        np.save(f"/mnt/HDD/val_data/{j}{i}_target.npy", f[int(len(f)*0.9):])
from verstack import LGBMTuner
import numpy as np
import pickle
import pandas as pd



amount = 100
trails = 1500

save = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default="0.0.0", help="Dataset version")
args = parser.parse_args()
version = args.version
        

if version[2] == "1":
    windowsize = 4
else:
    windowsize = 3

    
if version[0] != "0":
    import tensorflow_datasets as tfds
    data_dir = "../tensorflow_datasets"
    train_dataset = tfds.load(f'particle_data:{version}', data_dir=data_dir, split='train', as_supervised=True, shuffle_files=True)
    val_dataset = tfds.load(f'particle_data:{version}', data_dir=data_dir, split='val', as_supervised=True, shuffle_files=True)

    train_dataset = train_dataset.shuffle(buffer_size=50_000,reshuffle_each_iteration=False).take(amount)
    val_dataset = val_dataset.shuffle(buffer_size=50_000,reshuffle_each_iteration=False).take(amount)
    _X = [f[0] for f in train_dataset.as_numpy_iterator()]
    _y = [f[1] for f in train_dataset.as_numpy_iterator()]

    _X = np.stack(_X)
    _y = np.stack(_y)
    
else:
    data_dir = "/eos/user/m/mimodekj/data"
    if version[-1] == "1":
        _X,_y = np.load(f"{data_dir}/train_data_{windowsize}_backwards/train_input.npy"),np.load(f"{data_dir}/train_data_{windowsize}_backwards/train_target.npy")
        ID = np.argmax(np.load(f"{data_dir}/train_data_{windowsize}_backwards/train_id.npy"),axis=1)
        _X_test,_y_test = np.load(f"{data_dir}/test_data_{windowsize}_backwards/test_input.npy"),np.load(f"{data_dir}/test_data_{windowsize}_backwards/test_target.npy")
        ID_test = np.argmax(np.load(f"{data_dir}/test_data_{windowsize}_backwards/test_id.npy"),axis=1)
    else:
        _X,_y = np.load(f"{data_dir}/train_data_{windowsize}/train_input.npy"),np.load(f"{data_dir}/train_data_{windowsize}/train_target.npy")
        ID = np.argmax(np.load(f"{data_dir}/train_data_{windowsize}/train_id.npy"),axis=1)
        _X_test,_y_test = np.load(f"{data_dir}/test_data_{windowsize}/test_input.npy"),np.load(f"{data_dir}/test_data_{windowsize}/test_target.npy")
        ID_test = np.argmax(np.load(f"{data_dir}/test_data_{windowsize}/test_id.npy"),axis=1)
print("Data Loaded")

X0 = pd.DataFrame(_X[:500_000, :,0])
y0 = pd.Series(_y[:500_000,0])
X1 = pd.DataFrame(_X[:500_000, :,1])
y1 = pd.Series(_y[:500_000,1])
X2 = pd.DataFrame(_X[:500_000, :,2])
y2 = pd.Series(_y[:500_000,2])
XID = pd.DataFrame(ID[:500_000])

X0_test = pd.DataFrame(_X_test[:, :,0])
y0_test = pd.Series(_y_test[:,0])
X1_test = pd.DataFrame(_X_test[:, :,1])
y1_test = pd.Series(_y_test[:,1])
X2_test = pd.DataFrame(_X_test[:, :,2])
y2_test = pd.Series(_y_test[:,2])
XID_test = pd.DataFrame(ID_test)

y0_total = pd.Series(_y[:, 0])
y1_total = pd.Series(_y[:, 1])
y2_total = pd.Series(_y[:, 2])

if save:
    np.save(f"y0_{version}.npy",y0_test)
    np.save(f"y1_{version}.npy",y1_test)   
    np.save(f"y2_{version}.npy",y2_test)


X3 = pd.concat([X0,X1,X2,XID],axis=1)
X_total = np.hstack((_X[:, :,0],_X[:, :,1],_X[:, :,2],ID.reshape(-1,1)))
X0_total = np.hstack((_X[:, :,0],ID.reshape(-1,1)))
X1_total = np.hstack((_X[:, :,1],ID.reshape(-1,1)))
X2_total = np.hstack((_X[:, :,2],ID.reshape(-1,1)))
X3_test = pd.concat([X0_test,X1_test,X2_test, XID_test],axis=1)
X0 = pd.concat([X0,XID],axis=1)
X1 = pd.concat([X1,XID],axis=1)
X2 = pd.concat([X2,XID],axis=1)
X0_test = pd.concat([X0_test,XID_test],axis=1)
X1_test = pd.concat([X1_test,XID_test],axis=1)
X2_test = pd.concat([X2_test,XID_test],axis=1)

for tree_type in ['gbdt']:
    #tuned0 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    #tuned0.grid['num_leaves']['high'] = 2000
    #tuned0.grid['lambda_l1']['high'] = 100
    #tuned0.grid['lambda_l2']['high'] = 100
    #tuned0.fit(X0, y0)
    #tuned0.fit_optimized(X0_total, _y[:,0])
    #pickle.dump(tuned0, open(f"tuned0_{tree_type}_{version}.pkl", "wb"))

    #np.save(f"tuned0_predict_{tree_type}_{version}",tuned0.predict(X0_test))

    tuned01 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    tuned01.grid['num_leaves']['high'] = 2000
    tuned01.grid['lambda_l1']['high'] = 100
    tuned01.grid['lambda_l2']['high'] = 100
    tuned01.fit(X3, y0)
    tuned01.fit_optimized(X_total, _y[:,0])
    pickle.dump(tuned01, open(f"tuned01_{tree_type}_{version}.pkl", "wb"))

    np.save(f"tuned01_predict_{tree_type}_{version}",tuned01.predict(X3_test))

    tuned1 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    tuned1.grid['num_leaves']['high'] = 2000
    tuned1.grid['lambda_l1']['high'] = 100
    tuned1.grid['lambda_l2']['high'] = 100
    tuned1.fit(X1, y1)
    tuned1.fit_optimized(X1_total, _y[:,1])
    pickle.dump(tuned1, open(f"tuned1_{tree_type}_{version}.pkl", "wb"))

    np.save(f"tuned1_predict_{tree_type}_{version}",tuned1.predict(X1_test))

    tuned11 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    tuned11.grid['num_leaves']['high'] = 2000
    tuned11.grid['lambda_l1']['high'] = 100
    tuned11.grid['lambda_l2']['high'] = 100
    tuned11.fit(X3, y1)
    tuned11.fit_optimized(X_total, _y[:,1])
    pickle.dump(tuned11, open(f"tuned11_{tree_type}_{version}.pkl", "wb"))

    np.save(f"tuned11_predict_{tree_type}_{version}",tuned11.predict(X3_test))

    tuned2 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    tuned2.grid['num_leaves']['high'] = 2000
    tuned2.grid['lambda_l1']['high'] = 100
    tuned2.grid['lambda_l2']['high'] = 100
    tuned2.fit(X2, y2)
    tuned2.fit_optimized(X2_total, _y[:,2])
    pickle.dump(tuned2, open(f"tuned2_{tree_type}_{version}.pkl", "wb"))

    np.save(f"tuned2_predict_{tree_type}_{version}",tuned2.predict(X2_test))

    tuned21 = LGBMTuner(metric = 'mse', device_type = 'gpu', trials = trails, verbosity = 1, visualization = False, custom_metric = {"boosting": tree_type})
    tuned21.grid['num_leaves']['high'] = 2000
    tuned21.grid['lambda_l1']['high'] = 100
    tuned21.grid['lambda_l2']['high'] = 100
    tuned21.fit(X3, y2)
    tuned21.fit_optimized(X_total, _y[:,2])
    pickle.dump(tuned21, open(f"tuned21_{tree_type}_{version}.pkl", "wb"))

    np.save(f"tuned21_predict_{tree_type}_{version}",tuned21.predict(X3_test))

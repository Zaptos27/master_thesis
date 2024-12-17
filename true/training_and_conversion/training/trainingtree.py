from verstack import Stacker
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--nTracks", type=int, required=False, default=-1, help="force training for all epochs")
parser.add_argument("--gridsearch", type=int, default=50,)
parser.add_argument("--cluster", type=str, default="",)
parser.add_argument("--version", type=str, default="0.0.0", help="Dataset version")

#subparsers = parser.add_subparsers()
#bits = subparsers.add_parser('bits', help='number off bits for quantized network')
#bits.add_argument("--bits", type=int, default=8, help="Number of bits if using quantized network")

args = parser.parse_args()




if args.version[3] == "1":
    windowsize = 4
else:
    windowsize = 3


if args.version[0] != "0":
    train_dataset = tfds.load(f'particle_data:{args.version}', data_dir=data_dir, split='train', as_supervised=True, shuffle_files=True)
    val_dataset = tfds.load(f'particle_data:{args.version}', data_dir=data_dir, split='val', as_supervised=True, shuffle_files=True)

    train_dataset = train_dataset.repeat(args.epochs).shuffle(buffer_size=50_000,reshuffle_each_iteration=False)
    val_dataset = val_dataset.repeat(args.epochs).shuffle(buffer_size=50_000,reshuffle_each_iteration=False).take(args.batch*args.steps//10)


    train_dataset = train_dataset.batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(args.batch).prefetch(tf.data.experimental.AUTOTUNE)
else:
    if args.version[-1] == "1":
        train_dataset = tf.data.Dataset.from_tensor_slices((np.load(f"/eos/user/m/mimodekj/train_{windowsize}_backwards/1_input.npy"),np.load(f"/eos/user/m/mimodekj/train_{windowsize}_backwards/1_target.npy"))).batch(args.batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((np.load(f"/eos/user/m/mimodekj/val_{windowsize}_backwards/input.npy"),np.load(f"/eos/user/m/mimodekj/val_{windowsize}_backwards/target.npy"))).batch(args.batch)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((np.load(f"/eos/user/m/mimodekj/train_{windowsize}/1_input.npy"),np.load(f"/eos/user/m/mimodekj/train_{windowsize}/1_target.npy"))).batch(args.batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((np.load(f"/eos/user/m/mimodekj/val_{windowsize}/input.npy"),np.load(f"/eos/user/m/mimodekj/val_{windowsize}/target.npy"))).batch(args.batch)
    
    

print("Data Loaded")



stacker = Stacker(objective = 'regression',
                  auto = True,
                  auto_num_layers = 4,
                  metafeats = True,
                  epochs = args.epoch,
                  gridsearch_iterations = args.gridsearch,
                  stacking_feats_depth = 2,
                  include_X = False,
                  verbose = True)

X_transformed = stacker.fit_transform(X, y)

stacker.save_stacker(f"models/verstack{args.cluster}")



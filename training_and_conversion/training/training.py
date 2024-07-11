# from commonFunctions import rmNumpyPath
# rmNumpyPath()
import sys
import pickle 
import tensorflow as tf
import numpy as np
import os, sys
# from sklearn.model_selection import train_test_split
import argparse 
# import lossFunctions
import learningRates
from os.path import splitext
from packaging.version import Version
import tensorflow_datasets as tfds
import time


if Version(tf.__version__) > Version("2.16.0"): 
    version = True
else: 
    version = False

path = '/eos/user/m/mimodekj/master_thesis'
data_dir = "/eos/user/m/mimodekj/tensorflow_datasets"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ModelImporter():
    def __init__(self, modelType:str = "mlp",modelName:str = "mlp", windowSize:int=None, n_inputFeatures:int=None, n_outputFeatures:int=None):
        """
        import models as needed
        if using a new model, add the option to load it here
        """
        self.model = None
        if("mlp" in modelType.lower()): 
            from models_tf import MLP
            if modelName == "mlp": self.model = MLP.mlp(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_wide": self.model = MLP.mlp_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_deep": self.model = MLP.mlp_deep(windowSize, n_outputFeatures, dtype=dtype)
        elif("rnn" in modelType.lower()):
            from models_tf import RNN
            if modelName == "rnn": self.model = RNN.rnn(windowSize, n_inputFeatures, n_outputFeatures)
            elif modelName == "rnn_full_sequence": self.model = RNN.rnnFullSequence(n_inputFeatures, n_outputFeatures)
        elif("models" in modelType.lower()):
            from models_tf import models
            if modelName == "mlp_deep_wide": self.model = models.mlp_deep_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_conv": self.model = models.mlp_conv(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_conv_deep": self.model = models.mlp_conv_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_conv_wide": self.model = models.mlp_conv_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d": self.model = models.conv1d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d_deep": self.model = models.conv1d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d_wide": self.model = models.conv1d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv2d": self.model = models.conv2d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv2d_deep": self.model = models.conv2d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv2d_wide": self.model = models.conv2d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "depthwise_conv1d": self.model = models.depthwise_conv1d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "depthwise_conv1d_deep": self.model = models.depthwise_conv1d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "depthwise_conv1d_wide": self.model = models.depthwise_conv1d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv1d": self.model = models.separable_conv1d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv1d_deep": self.model = models.separable_conv1d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv1d_wide": self.model = models.separable_conv1d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv2d": self.model = models.separable_conv2d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv2d_deep": self.model = models.separable_conv2d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "separable_conv2d_wide": self.model = models.separable_conv2d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_piecewise": self.model = models.mlp_piecewise(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_piecewise_deep": self.model = models.mlp_piecewise_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_piecewise_wide": self.model = models.mlp_piecewise_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_piecewise_deep_wide": self.model = models.mlp_piecewise_deep_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "convlstm1d": self.model = models.convlstm1d(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "convlstm1d_deep": self.model = models.convlstm1d_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "convlstm1d_wide": self.model = models.convlstm1d_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "lstm": self.model = models.lstm(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "lstm_deep": self.model = models.lstm_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "lstm_wide": self.model = models.lstm_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_repeat": self.model = models.mlp_repeat(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_repeat_deep": self.model = models.mlp_repeat_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "mlp_repeat_wide": self.model = models.mlp_repeat_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "rnn": self.model = models.rnn(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "rnn_deep": self.model = models.rnn_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "rnn_wide": self.model = models.rnn_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "gru": self.model = models.gru(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "gru_deep": self.model = models.gru_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "gru_wide": self.model = models.gru_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "stacked_rnn": self.model = models.stacked_rnn(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "stacked_rnn_deep": self.model = models.stacked_rnn_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "stacked_rnn_wide": self.model = models.stacked_rnn_wide(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "stacked_gru": self.model = models.stacked_gru(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "stacked_gru_deep": self.model = models.stacked_gru_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d_rnn": self.model = models.conv1d_rnn(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d_rnn_deep": self.model = models.conv1d_rnn_deep(windowSize, n_outputFeatures, dtype=dtype)
            elif modelName == "conv1d_rnn_wide": self.model = models.conv1d_rnn_wide(windowSize, n_outputFeatures, dtype=dtype)


        
tf.keras.backend.clear_session()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Start time
start = time.time()
print("Start time: ", start)

parser = argparse.ArgumentParser()
parser.add_argument("--windowSize", type=int, default=3, help="window size of algorithm")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--batch", type=int, default=2048, help="batch size")
parser.add_argument("--retrain", action='store_true', help="retrain an already existing model")
parser.add_argument("--forceAllEpochs", action='store_true', help="force training for all epochs")
parser.add_argument("--nTracks", type=int, required=False, default=-1, help="force training for all epochs")
parser.add_argument("--dtype", type=str, default="float32", help="data type for training")
parser.add_argument("--modelName", type=str, default="mlp", required=True, help="Name of the trained model")
parser.add_argument("--modelType", type=str, default="MLP",)

#subparsers = parser.add_subparsers()
#bits = subparsers.add_parser('bits', help='number off bits for quantized network')
#bits.add_argument("--bits", type=int, default=8, help="Number of bits if using quantized network")

args = parser.parse_args()

if args.dtype == "float32":
    dtype = tf.float32
elif args.dtype == "float64":
    dtype = tf.float64
elif args.dtype == "float16":
    dtype = tf.float16
elif args.dtype == "float8":
    dtype = tf.float8
else:
    raise ValueError("dtype not recognized")

train_dataset = tfds.load('particle_data', data_dir=data_dir, split='train', as_supervised=True).batch(args.batch)
val_dataset = tfds.load('particle_data', data_dir=data_dir, split='val', as_supervised=True).batch(args.batch)

n_inputFeatures = 3#,inputs_train.shape[-1]
n_outputFeatures = 3#targets_train.shape[-1]

#train_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, targets_train)).batch(args.batch)
#val_dataset = tf.data.Dataset.from_tensor_slices((inputs_val, targets_val)).batch(args.batch)


print("creating model:", args.modelName)

if args.retrain:
    model = tf.keras.models.load_model(path+'/models/tensorflow/{}'.format(args.modelName))
    print("loaded pre-trained model")

else:
    modelImporter = ModelImporter(args.modelType, args.modelName, args.windowSize, n_inputFeatures, n_outputFeatures)
    model = modelImporter.model
    print("model: ", model)

checkpoint_filepath = 'checkpoints/' + args.modelName
os.system("mkdir -vp "+checkpoint_filepath)

if (args.retrain): model.load_weights(checkpoint_filepath)

if version:
    checkpoint_filepath = checkpoint_filepath + '/model.weights.h5'

if args.forceAllEpochs:
    patience = args.epochs
else:
    patience = 50
callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True),
			tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)]

training_history = {"loss":[], "val_loss":[]}


warmup_steps =  args.batch * args.epochs // 5 # 1/5 of the training batches will be warmup steps
learning_rate = learningRates.CustomSchedule(64.0, warmup_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss='mse'
model.compile(optimizer=optimizer, loss=loss)


history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks)



print(model.summary())
training_history['loss'] += history.history['loss']
training_history["val_loss"] += history.history['val_loss']

model.load_weights(checkpoint_filepath)

# save the best model only
from shutil import rmtree
modelPath = "../models/tensorflow/" +args.modelName
if version:
    modelPath = modelPath + ".h5"
    if os.path.exists(modelPath): os.remove(modelPath)
else:
    if os.path.exists(modelPath): rmtree(modelPath)

#saveDir = "../models/tensorflow/"
tf.keras.models.save_model(model, modelPath, include_optimizer=False)

# print("python -m tf2onnx.convert --saved-model " +saveDir+ args.modelName+ " --output "+saveDir+"model.onnx")
# os.system("python -m tf2onnx.convert --saved-model "+saveDir + args.modelName + " --output "+saveDir+args.modelName+".onnx")

#save training info to plot later
os.system("mkdir -vp loss_info/")
with open("loss_info/"+args.modelName + ".pkl", "wb") as f:
    pickle.dump(training_history, f)

print("Training complete")

# End time
end = time.time()
print("End time: ", end)
print("Time taken: ", end-start)


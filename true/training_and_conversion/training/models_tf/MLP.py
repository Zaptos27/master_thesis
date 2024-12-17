import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D
from tensorflow.keras.models import Sequential

"""
Define here any and all variations of MLP models 
"""

def mlp(window_size=3, n_outputFeatures=3, activation='linear', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
    model  = Sequential([
        Input((window_size,n_outputFeatures), dtype=dtype),
        Flatten(), 
        Dense(32, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_outputFeatures, activation=activation, dtype=dtype),
    ])
    return model

def mlp_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
    model  = Sequential([
        Input((window_size,n_features), dtype=dtype),
        Flatten(), 
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def mlp_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
    model  = Sequential([
        Input((window_size,n_features), dtype=dtype),
        Flatten(), 
        Dense(32, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model


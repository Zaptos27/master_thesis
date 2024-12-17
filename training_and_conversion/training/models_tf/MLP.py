import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Reshape
from tensorflow.keras.models import Sequential

"""
Define here any and all variations of MLP models 
"""

#def mlp(window_size=3, n_outputFeatures=3, activation='linear', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
#    model  = Sequential([
#        Input((window_size,n_outputFeatures), dtype=dtype),
#        Input(1, dtype=dtype),
#        Flatten(), 
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_outputFeatures, activation=activation, dtype=dtype),
#    ])
#    return model


def mlp(window_size=3, n_outputFeatures=3, activation='linear', dtype=tf.float32):
    input1 = Input((window_size,n_outputFeatures), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Flatten()(input1)
    x = Concatenate()([x, input2])
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(n_outputFeatures, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=x)

    return model

#def mlp_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
#    model  = Sequential([
#        Input((window_size,n_features), dtype=dtype),
#        Input(1, dtype=dtype),
#        Flatten(), 
#        Dense(128, activation='relu', dtype=dtype),
#        Dense(256, activation='relu', dtype=dtype),
#        Dense(128, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def mlp_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Flatten()(input1)
    x = Concatenate()([x, input2])
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=x)

    return model


#def mlp_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    #window_size refers to the number of hits included in a data sample used for prediction
    #n_feautres refers to the number of features per hit, ex. 3 for x,y,z
#    model  = Sequential([
#        Input((window_size,n_features), dtype=dtype),
#        Input(1, dtype=dtype),
#        Flatten(), 
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def mlp_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Flatten()(input1)
    x = Concatenate()([x, input2])
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=x)

    return model



#def mlp_big(window_size=3, n_features=3, n_detector=216):
#    """
#    Extrapolator takes in window of hits and an expected next hit detector ID.
#    n_detector is the maximum class label (number of classes)
#    """
#    input_hits = tf.keras.Input(shape=(window_size, n_features)) # track hits
#    input_detector = tf.keras.Input(shape=(n_detector,)) # OHE detector id 
#    # x = tf.keras.layers.Flatten()(input_hits)
#    x = Reshape( (window_size*n_features,) )(input_hits)
#    merge = tf.keras.layers.Concatenate(axis=1)([x, input_detector])
#    x = tf.keras.layers.Dense(256, activation='relu')(merge)
#    x = tf.keras.layers.Dense(128, activation='relu')(x)
#    x = tf.keras.layers.Dense(64, activation='relu')(x)
#    x = tf.keras.layers.Dense(32, activation='relu')(x)
#    output = tf.keras.layers.Dense(n_features, activation='tanh')(x)
#    model = tf.keras.Model(inputs=[input_hits, input_detector], outputs=output)
#
#    return model


def mlp_big(window_size=3, n_features=3, n_detector=216, activation= 'tanh', dtype = tf.float32):
    """
    Extrapolator takes in window of hits and an expected next hit detector ID.
    n_detector is the maximum class label (number of classes)
    """
    input_hits = tf.keras.Input(shape=(window_size, n_features), dtype=dtype, name="input1") # track hits
    input_detector = tf.keras.Input(shape=(n_detector,), dtype=dtype, name="input2") # OHE detector id 
    x = Reshape( (window_size*n_features,) )(input_hits)
    merge = Concatenate(axis=1)([x, input_detector])
    x = Dense(256, activation='relu', dtype=dtype)(merge)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input_hits, input_detector], outputs=output)

    return model

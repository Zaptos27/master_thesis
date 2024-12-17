import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, Conv2D, DepthwiseConv2D, DepthwiseConv1D, SeparableConv1D, SeparableConv2D, Concatenate, ConvLSTM1D, LSTM, RepeatVector, SimpleRNN, StackedRNNCells, SimpleRNNCell, GRU, GRUCell, RNN
from tensorflow.keras.models import Sequential

def mlp_small(window_size=3, n_outputFeatures=3, activation='tanh', dtype=tf.float32):
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


#def mlp_deep_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        Dense(128, activation='relu', dtype=dtype),
#        Dense(256, activation='relu', dtype=dtype),
#        Dense(128, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def mlp_deep_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Flatten()(input1)
    x = Concatenate()([x, input2])
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    return model


#def mlp_conv(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype)
#    ])
#    return model

def mlp_conv(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(input1)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    return model

def mlp_conv_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def mlp_conv_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(64, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model


#def conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
#        Conv1D(64, 1, activation='relu', dtype=dtype),
#        Conv1D(32, 1, activation='relu', dtype=dtype),
#        Flatten(),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(input1)
    x = Conv1D(64, 1, activation='relu', dtype=dtype)(x)
    x = Conv1D(32, 1, activation='relu', dtype=dtype)(x)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model



def conv1d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Conv1D(64, 1, activation='relu', dtype=dtype),
        Conv1D(64, 1, activation='relu', dtype=dtype),
        Conv1D(64, 1, activation='relu', dtype=dtype),
        Conv1D(32, 1, activation='relu', dtype=dtype),
        Flatten(),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def conv1d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Conv1D(128, 1, activation='relu', dtype=dtype),
        Conv1D(256, 1, activation='relu', dtype=dtype),
        Conv1D(128, 1, activation='relu', dtype=dtype),
        Flatten(),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def conv2d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = tf.expand_dims(input1, 3)
    x = Conv2D(32, (2, 2), activation='relu', dtype=dtype)(x)
    x = Conv2D(64, (2, 2), activation='relu', dtype=dtype)(x)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model

def conv2d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = tf.expand_dims(input1, 3)
    x = Conv2D(32, (2, 2), activation='relu', dtype=dtype)(x)
    x = Conv2D(64, (2, 2), activation='relu', dtype=dtype)(x)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model

def conv2d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = tf.expand_dims(input1, 3)
    x = Conv2D(128, (2, 2), activation='relu', dtype=dtype)(x)
    x = Conv2D(256, (2, 2), activation='relu', dtype=dtype)(x)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model


#def depthwise_conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        DepthwiseConv1D(2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def depthwise_conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = DepthwiseConv1D(2, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(input1)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model

def depthwise_conv1d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        DepthwiseConv1D(2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def depthwise_conv1d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        DepthwiseConv1D(2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

#def separable_conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        SeparableConv1D(32, 2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def separable_conv1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = SeparableConv1D(32, 2, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(input1)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model



def separable_conv1d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        SeparableConv1D(32, 2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def separable_conv1d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        SeparableConv1D(32, 2, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

#def separable_conv2d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features, 1), dtype=dtype),
#        SeparableConv2D(32, (2, 2), activation='relu', input_shape=(window_size, n_features, 1), dtype=dtype),
#        Flatten(),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#   return model

def separable_conv2d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input(shape=(window_size, n_features, 1), dtype=dtype, name="input1")
    input2 = Input(shape=(216,), dtype=dtype, name="input2")
    x = SeparableConv2D(32, (2, 2), activation='relu', input_shape=(window_size, n_features, 1), dtype=dtype)(input1)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model


def separable_conv2d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features, 1), dtype=dtype),
        SeparableConv2D(32, (2, 2), activation='relu', input_shape=(window_size, n_features, 1), dtype=dtype),
        Flatten(dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def separable_conv2d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features, 1), dtype=dtype),
        SeparableConv2D(32, (2, 2), activation='relu', input_shape=(window_size, n_features, 1), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def mlp_piecewise(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input(shape=(window_size, n_features),dtype=dtype)
    input2 = Input(shape=(216,), dtype=dtype)
    x = Flatten()(tf.slice(input1, [0, 0, 0], [-1, -1, 1]))
    y = Flatten()(tf.slice(input1, [0, 0, 1], [-1, -1, 1]))
    z = Flatten()(tf.slice(input1, [0, 0, 2], [-1, -1, 1]))
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    out = Concatenate()([x, y, z, input2])
    out = Dense(128, activation='relu', dtype=dtype)(out)
    out = Dense(n_features, activation=activation, dtype=dtype)(out)
    model = tf.keras.Model(inputs=[input1, input2], outputs=out)
    return model


def mlp_piecewise2(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input(shape=(window_size, n_features),dtype=dtype)
    input2 = Input(shape=(216,), dtype=dtype)
    x = Flatten()(tf.slice(input1, [0, 0, 0], [-1, -1, 1]))
    y = Flatten()(tf.slice(input1, [0, 0, 1], [-1, -1, 1]))
    z = Flatten()(tf.slice(input1, [0, 0, 2], [-1, -1, 1]))
    x = Concatenate()([x, input2])
    y = Concatenate()([y, input2])
    z = Concatenate()([z, input2])
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    out = Concatenate()([x, y, z])
    out = Dense(128, activation='relu', dtype=dtype)(out)
    out = Dense(n_features, activation=activation, dtype=dtype)(out)
    model = tf.keras.Model(inputs=[input1, input2], outputs=out)
    return model

def mlp_piecewise_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    inputs = Input(shape=(window_size, n_features),dtype=dtype)
    x = Flatten()(tf.slice(inputs, [0, 0, 0], [-1, -1, 1]))
    y = Flatten()(tf.slice(inputs, [0, 0, 1], [-1, -1, 1]))
    z = Flatten()(tf.slice(inputs, [0, 0, 2], [-1, -1, 1]))
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    out = Concatenate()([x, y, z])
    out = Dense(64, activation='relu', dtype=dtype)(out)
    out = Dense(n_features, activation=activation, dtype=dtype)(out)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model

def mlp_piecewise_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    inputs = Input(shape=(window_size, n_features),dtype=dtype)
    x = Flatten()(tf.slice(inputs, [0, 0, 0], [-1, -1, 1]))
    y = Flatten()(tf.slice(inputs, [0, 0, 1], [-1, -1, 1]))
    z = Flatten()(tf.slice(inputs, [0, 0, 2], [-1, -1, 1]))
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    y = Dense(128, activation='relu', dtype=dtype)(y)
    z = Dense(128, activation='relu', dtype=dtype)(z)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    y = Dense(256, activation='relu', dtype=dtype)(y)
    z = Dense(256, activation='relu', dtype=dtype)(z)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    y = Dense(128, activation='relu', dtype=dtype)(y)
    z = Dense(128, activation='relu', dtype=dtype)(z)
    out = Concatenate()([x, y, z])
    out = Dense(128, activation='relu', dtype=dtype)(out)
    out = Dense(n_features, activation=activation, dtype=dtype)(out)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model

def mlp_piecewise_deep_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    inputs = Input(shape=(window_size, n_features),dtype=dtype)
    x = Flatten()(tf.slice(inputs, [0, 0, 0], [-1, -1, 1]))
    y = Flatten()(tf.slice(inputs, [0, 0, 1], [-1, -1, 1]))
    z = Flatten()(tf.slice(inputs, [0, 0, 2], [-1, -1, 1]))
    x = Dense(32, activation='relu', dtype=dtype)(x)
    y = Dense(32, activation='relu', dtype=dtype)(y)
    z = Dense(32, activation='relu', dtype=dtype)(z)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    y = Dense(128, activation='relu', dtype=dtype)(y)
    z = Dense(128, activation='relu', dtype=dtype)(z)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    y = Dense(256, activation='relu', dtype=dtype)(y)
    z = Dense(256, activation='relu', dtype=dtype)(z)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    y = Dense(256, activation='relu', dtype=dtype)(y)
    z = Dense(256, activation='relu', dtype=dtype)(z)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    y = Dense(128, activation='relu', dtype=dtype)(y)
    z = Dense(128, activation='relu', dtype=dtype)(z)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    y = Dense(64, activation='relu', dtype=dtype)(y)
    z = Dense(64, activation='relu', dtype=dtype)(z)
    out = Concatenate()([x, y, z])
    out = Dense(128, activation='relu', dtype=dtype)(out)
    out = Dense(n_features, activation=activation, dtype=dtype)(out)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


def convlstm1d(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input = Input(shape=(window_size, n_features), dtype=dtype)
    x = tf.expand_dims(input, 3)
    x = ConvLSTM1D(32, 3, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def convlstm1d_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input = Input(shape=(window_size, n_features), dtype=dtype)
    x = tf.expand_dims(input, 3)
    x = ConvLSTM1D(32, 3, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def convlstm1d_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input = Input(shape=(window_size, n_features), dtype=dtype)
    x = tf.expand_dims(input, 3)
    x = ConvLSTM1D(32, 3, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', dtype=dtype)(x)
    x = Dense(256, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

#def lstm(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        LSTM(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def lstm(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = LSTM(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype)(input1)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model



def lstm_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        LSTM(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def lstm_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        LSTM(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

#def mlp_repeat(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
#    model = Sequential([
#        Input(shape=(window_size, n_features), dtype=dtype),
#        Flatten(),
#        RepeatVector(3),
#        Flatten(),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(64, activation='relu', dtype=dtype),
#        Dense(32, activation='relu', dtype=dtype),
#        Dense(n_features, activation=activation, dtype=dtype),
#    ])
#    return model

def mlp_repeat(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    input1 = Input((window_size,n_features), dtype=dtype,  name="input1")
    input2 = Input((216,), dtype=dtype, name="input2")
    x = Flatten()(input1)
    x = RepeatVector(3)(x)
    x = Flatten()(x)
    x = Concatenate()([x, input2])
    x = Dense(32, activation='relu', dtype=dtype)(x)
    x = Dense(64, activation='relu', dtype=dtype)(x)
    x = Dense(32, activation='relu', dtype=dtype)(x)
    output = Dense(n_features, activation=activation, dtype=dtype)(x)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model


def mlp_repeat_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        RepeatVector(3),
        Flatten(),
        Dense(32, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def mlp_repeat_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        RepeatVector(3),
        Flatten(),
        Dense(32, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def rnn(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        #Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def rnn_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def rnn_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def gru(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        GRU(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def gru_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        GRU(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def gru_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        GRU(32, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def stacked_rnn(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        RNN([SimpleRNNCell(32, activation='relu') for _ in range(3)], dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def stacked_rnn_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        RNN([SimpleRNNCell(32, activation='relu'), SimpleRNNCell(32, activation='relu')], dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def stacked_rnn_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        RNN([SimpleRNNCell(32, activation='relu') for _ in range(3)], dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def stacked_gru(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        RNN([GRUCell(32, activation='relu'), GRUCell(32, activation='relu')], dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def stacked_gru_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        RNN([GRUCell(32, activation='relu'), GRUCell(32, activation='relu')], dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model


def conv1d_rnn(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def conv1d_rnn_deep(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', dtype=dtype),
        Flatten(),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(64, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(32, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

def conv1d_rnn_wide(window_size=3, n_features=3, activation='tanh', dtype=tf.float32):
    model = Sequential([
        Input(shape=(window_size, n_features), dtype=dtype),
        Conv1D(32, 1, activation='relu', input_shape=(window_size, n_features), dtype=dtype),
        SimpleRNN(32, activation='relu', dtype=dtype),
        Flatten(),
        Dense(128, activation='relu', dtype=dtype),
        Dense(256, activation='relu', dtype=dtype),
        Dense(128, activation='relu', dtype=dtype),
        Dense(n_features, activation=activation, dtype=dtype),
    ])
    return model

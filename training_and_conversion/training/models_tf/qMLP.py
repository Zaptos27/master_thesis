import tensorflow as tf
from qkeras import QDense, QActivation

def qmlp(window_size=3, n_features=3, bits=8):

    input_layer = tf.keras.Input(shape=(window_size,n_features))

    x = tf.keras.layers.Flatten()(input_layer)
    x = QDense(64,
               kernel_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits),
               bias_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits))(x)
    x = QActivation('quantized_relu({},0)'.format(bits))(x)
    x = QDense(32,
               kernel_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits),
               bias_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits))(x)
    x = QActivation('quantized_relu({},0)'.format(bits))(x)
    x = QDense(1,
               kernel_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits),
               bias_quantizer = 'quantized_bits({},0,alpha=1)'.format(bits))(x)

    output = tf.keras.layers.Activation('tanh')(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])

    return model
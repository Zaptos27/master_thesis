import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GRU, Input, Bidirectional, Masking
from tensorflow.keras.models import Sequential

def rnn(window_size=3, n_inputFeatures=20,n_outputFeatures=3, dtype = tf.float32):
    inputs = Input(shape=(window_size,n_inputFeatures))
    #x = Masking(mask_value=0)(inputs)
    x = GRU(32, return_sequences=True, dtype=dtype)(inputs)
    x = GRU(32, return_sequences=False, dtype=dtype)(x)

    outputs = Dense(n_outputFeatures, activation='tanh', name='output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def biRNN(window_size=3, n_inputFeatures=48,n_outputFeatures=3, masking=False):
    inputs = Input(shape=(window_size,n_inputFeatures))
    if masking:
        x = Masking(mask_value=0)(inputs)
        x = Bidirectional(GRU(64, return_sequences=True))(x)
    else:
        x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = Dense(64, activation='selu')(x)
    outputs = Dense(n_outputFeatures, activation='tanh', name='output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def rnnFullSequence(n_inputFeatures=3, n_outputFeatures=3, masking=False, dtype=tf.float32):
    """
    input data must be carefully padded. The input sequence length will be equal to the output sequence length
    Masking layer is buggy in ONNX
    """
    model = tf.keras.models.Sequential()
    if masking:
        model.add(tf.keras.layers.Masking())
    model.add(GRU(32, return_sequences=True, dtype=dtype))
    model.add(GRU(32, return_sequences=True, dtype=dtype))
    model.add(GRU(3, return_sequences=True, dtype=dtype))
    return model

class GaussianRNN(tf.keras.Model):
    """
    To be trained using gaus_llh_loss.
    Requires its own training script due to the custom training loop

    see https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html for
    implimentation using tf probabilistic layers
    """
    def __init__(self, n_inputHits, n_inputFeatures=3, n_outputFeatures=5):
        #output feature_1, feature_2, sigma_1, sigma_2, corr_12
        super().__init__()
        self.n_inputHits = n_inputHits
        
        self.inputs = Input(shape=(n_inputHits,n_inputFeatures))
        self.gru1 = GRU(32, return_sequences=True)
        self.gru2 = GRU(32, return_sequences=True)
        self.out = Dense(5)
        
    def call(self, inputs):
        shape = tf.shape(inputs)
        x = self.gru1(inputs)
        x = self.gru2(x)
        outputs = self.out(x)
        outputs = tf.reshape(outputs, (-1, 5))
        mean = outputs[:,:2]
        var = tf.math.exp(outputs[:,2:4])
        var = tf.math.sqrt(var)
        correlations = tf.math.tanh(outputs[:,4:])
        correlations = tf.reshape(correlations, (shape[0]*shape[1],1,1))
        identity = tf.eye(2, batch_shape=[shape[0]*shape[1]],dtype=tf.float32)
        
        corr_matrix = tf.cast(tf.math.equal(identity,0), tf.float32) * correlations
        corr_matrix = identity+corr_matrix
        
        covs = tf.matmul(var[:, :,None], var[:, None, :])
        covs = covs * corr_matrix
        mean = tf.reshape(mean, (shape[0], shape[1], 2))
        covs = tf.reshape(covs, (shape[0], shape[1],2,2))
        return mean, covs
    
    def train_step(self, inputs):
        inputs, targets = inputs
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.compiled_loss(outputs, targets)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}
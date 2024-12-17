
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def gaussianMLP():
    """
    requires negloglik = lambda y, p_y: -p_y.log_prob(y)
    as the loss function
    """
    model = tf.Sequential([
    tf.keras.layers.Dense(32),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
    ])
    
    return model


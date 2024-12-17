
import tensorflow as tf
#import tensorflow_addons as tfa

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000.0):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    self.warmup_steps = tf.cast(self.warmup_steps, tf.float32)

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class OCP(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=4000, lr_min=1e-4, lr_max=5e-2 ):
    super(OCP, self).__init__()
    self.warmup_steps = warmup_steps
    self.slope1 = (lr_max - lr_min) / (warmup_steps / 2) + lr_min
    self.slope2 = -(lr_max - lr_min) / (warmup_steps / 2) + lr_min + lr_max
  def __call__(self, step):
    arg1 = step*self.slope1
    arg2 = step *self.slope2

    return tf.math.minimum(arg1, arg2)

# def CLR(lr_min, lr_max, step_size):
#   clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=lr_min,
#       maximal_learning_rate=lr_max,
#       scale_fn=lambda x: 1/(2.**(x-1)),
#       step_size=step_size
#   )
#   return tf.keras.optimizers.SGD(clr)

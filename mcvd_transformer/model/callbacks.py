from keras.callbacks import LearningRateScheduler, Callback  # type: ignore
from tensorflow.keras import backend as K  # type: ignore

import tensorflow as tf


def lr_scheduler(
    epoch,
    lr,
    warmup_epochs=25,
    decay_epochs=175,
    initial_lr=1e-5,
    base_lr=1e-3,
    min_lr=5e-5,
):

    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs + decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


class AdaptiveLossWeight(Callback):

    def __init__(self, alpha, beta, warmup_epochs=30, decay_epochs=200):
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = tf.cast(warmup_epochs, tf.float32)
        self.decay_epochs = tf.cast(decay_epochs, tf.float32)
        self.initial_alpha = K.get_value(alpha)

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):

        if epoch > self.warmup_epochs:
            pct = 1 - ((epoch - self.warmup_epochs) / self.decay_epochs)
            K.set_value(self.alpha, max(self.initial_alpha * pct, 0))
            K.set_value(self.beta, 1 - K.get_value(self.alpha))

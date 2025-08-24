from keras.layers import Dense, Multiply  # type: ignore
from keras.layers import Flatten, Input, concatenate  # type: ignore
from keras.models import Model  # type: ignore
from keras.optimizers import Adam  # type: ignore

from .layers import TransformerBlock
from .metrics import (
    weighted_loss,
    custom_mse,
    custom_mape,
    max_squared_loss,
    r2_score,
    max_loss,
    dummy_loss
)
import tensorflow as tf


def create_model(input_topology_shape, input_topology_constants_shape):

    # Combined Part of the ANN
    input_topology = Input(shape=input_topology_shape)
    x = TransformerBlock(key_dim=input_topology_shape[1], num_heads=16, ff_dim=32)(
        input_topology, training=True
    )
    x = TransformerBlock(key_dim=input_topology_shape[1], num_heads=16, ff_dim=32)(
        x, training=True
    )
    x = Flatten()(x)

    input_num = Input(shape=(input_topology_constants_shape,))
    x = concatenate([x, input_num])
    x = Dense(400, activation="relu")(x)
    x = Dense(800, activation="relu")(x)
    x = Dense(1600, activation="relu")(x)

    # Shape Part of the ANN
    x1 = Dense(2 * 810, activation="relu")(x)
    x1 = Dense(810, name="shape")(x1)

    # Maximum Value Part of the ANN
    x2 = Dense(1600, activation="relu")(x)
    x2 = Dense(800, activation="relu")(x2)
    x2 = Dense(400, activation="relu")(x2)
    x2 = Dense(1, name="max")(x2)

    x3 = Multiply(name="cir")([x1, x2])

    alpha = tf.Variable(1.0, dtype=tf.float32)
    beta = tf.Variable(0.0, dtype=tf.float32)

    model = Model(inputs=[input_topology, input_num], outputs=[x1, x2, x3])
    model.compile(
        loss=[
            weighted_loss(custom_mse, alpha),
            weighted_loss(custom_mape, alpha),
            weighted_loss(max_squared_loss, beta),
        ],
        optimizer=Adam(),
        metrics=[r2_score, max_loss, dummy_loss],
    )

    return model, alpha, beta

from tensorflow.keras import backend as K  # type: ignore
from keras.losses import mean_absolute_percentage_error, mean_squared_error  # type: ignore
import tensorflow as tf

K.set_epsilon(1)


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def custom_mape(y_true, y_pred):
    return (mean_absolute_percentage_error(y_true, y_pred) / 100) ** 2


def custom_mse(y_true, y_pred):

    # Calculate the pdf of the tensor
    y_pred_pdf_neg = y_pred[:, :-1] - y_pred[:, 1:]

    # calculating squared difference between target and predicted values
    loss = K.square(tf.nn.relu(y_pred_pdf_neg))  # (batch_size, 2)

    # summing both loss values along batch dimension
    loss = K.mean(loss, axis=1)  # (batch_size,)

    return mean_squared_error(y_true, y_pred) + 0.0001 * loss


def dummy_loss(y_true, y_pred):
    return 0


def max_squared_loss(y_true, y_pred):
    return K.max(
        K.square(y_true / y_true[:, -1, None] - y_pred / y_true[:, -1, None]), axis=-1
    )


def max_loss(y_true, y_pred):
    return K.max(
        K.abs(y_true / y_true[:, -1, None] - y_pred / y_true[:, -1, None]), axis=-1
    )


def der_max_squared_error(y_true, y_pred):
    squared_error = K.square(
        y_true / y_true[:, -1, None] - y_pred / y_true[:, -1, None]
    )
    return K.sum(tf.nn.softmax(squared_error) * squared_error, axis=-1)


def derivative_max_loss(y_true, y_pred):

    window_size = 2

    y_true_n = y_true / y_true[:, -1, None]
    y_pred_n = y_pred / y_true[:, -1, None]

    der_y_pred = y_pred_n[:, 1:] - y_pred_n[:, :-1]
    der_y_true = y_true_n[:, 1:] - y_true_n[:, :-1]

    abs_err = K.abs(der_y_pred - der_y_true)

    windowed_abs_err = tf.vectorized_map(
        lambda i: abs_err[:, i : i + window_size],
        tf.range(abs_err.shape[1] - window_size + 1),
    )

    total_windowed_abs_err = K.sum(windowed_abs_err, axis=2)

    return K.max(total_windowed_abs_err, axis=1)


def weighted_loss(loss_func, loss_weight):

    def loss(y_true, y_pred):
        return loss_weight * loss_func(y_true, y_pred)

    return loss

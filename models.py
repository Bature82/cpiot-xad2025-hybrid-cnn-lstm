"""One architecture, three stream configurations.

"fused" is the proposed dual-stream model: a 1-D CNN over the network
telemetry stream and a stacked LSTM over the physical-process stream, joined by
concatenation and predicting the next timestep of both streams. "net_only" and
"sen_only" are the single-stream ablations, trained under an identical protocol
so that any difference is attributable to the streams themselves.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Conv1D, Dense, Dropout,
    GlobalMaxPooling1D, Input, LSTM,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class MemCallback(tf.keras.callbacks.Callback):
    """Print RSS every few epochs so a crash mid-fit is visible in the log."""

    def __init__(self, mem_fn):
        super().__init__()
        self.mem_fn = mem_fn

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.mem_fn(f"  fit epoch {epoch}")


def build_model(mode, net_shape, sen_shape, lr, clipnorm, delta, drop):
    huber = Huber(delta=delta)
    if mode == "fused":
        net_in = Input(shape=net_shape, name="network_input")
        x = Conv1D(96, 3, activation="relu", padding="same")(net_in)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        x = Conv1D(48, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)

        sen_in = Input(shape=sen_shape, name="sensor_input")
        y = LSTM(96, return_sequences=True)(sen_in)
        y = Dropout(drop)(y)
        y = LSTM(48)(y)
        y = BatchNormalization()(y)

        z = Concatenate()([x, y])
        out_net = Dense(net_shape[-1], name="network_pred")(z)
        out_sen = Dense(sen_shape[-1], name="sensor_pred")(z)
        m = Model([net_in, sen_in], [out_net, out_sen])
        m.compile(optimizer=Adam(learning_rate=lr, clipnorm=clipnorm),
                  loss={"network_pred": huber, "sensor_pred": huber})
    elif mode == "net_only":
        net_in = Input(shape=net_shape, name="network_input")
        x = Conv1D(96, 3, activation="relu", padding="same")(net_in)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        x = Conv1D(48, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        out_net = Dense(net_shape[-1], name="network_pred")(x)
        m = Model(net_in, out_net)
        m.compile(optimizer=Adam(learning_rate=lr, clipnorm=clipnorm), loss=huber)
    elif mode == "sen_only":
        sen_in = Input(shape=sen_shape, name="sensor_input")
        y = LSTM(96, return_sequences=True)(sen_in)
        y = Dropout(drop)(y)
        y = LSTM(48)(y)
        y = BatchNormalization()(y)
        out_sen = Dense(sen_shape[-1], name="sensor_pred")(y)
        m = Model(sen_in, out_sen)
        m.compile(optimizer=Adam(learning_rate=lr, clipnorm=clipnorm), loss=huber)
    else:
        raise ValueError(mode)
    return m

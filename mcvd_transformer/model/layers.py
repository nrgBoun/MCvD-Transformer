from keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Layer  # type: ignore
from keras import Sequential
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings


class TransformerBlock(Layer):

    def __init__(self, key_dim, num_heads, ff_dim, rate=0.1, **kwargs):

        self.key_dim = key_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(key_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):

        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config

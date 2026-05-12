import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Optional, Tuple


def build_rnn_decoder_pre_inject(
    vocab_size: int,
    embed_dim: int = 256,
    rnn_units: int = 256,
    feature_dim: int = 2048,
    num_rnn_layers: int = 1,
    max_caption_len: int = 30,
    rnn_type: str = "rnn",
    dropout: float = 0.0,
) -> Model:
    rnn_type = rnn_type.lower()

    cnn_input = keras.Input(shape=(feature_dim,), name="cnn_input")  # (B, F)
    word_input = keras.Input(shape=(None,), dtype="int32", name="word_input")  # (B, T)

    x_img = layers.Dense(embed_dim, name="image_projection")(cnn_input)
    x_img = layers.Reshape((1, embed_dim), name="image_token")(x_img)  # (B, 1, E)

    x_words = layers.Embedding(vocab_size, embed_dim, name="embedding")(word_input)

    x = layers.Concatenate(axis=1, name="concat_img_words")([x_img, x_words])

    for i in range(num_rnn_layers):
        return_seq = True
        if rnn_type == "lstm":
            rnn_layer = layers.LSTM(
                rnn_units,
                return_sequences=return_seq,
                dropout=dropout,
                name=f"rnn_layer_{i}",
            )
        else:
            rnn_layer = layers.SimpleRNN(
                rnn_units,
                return_sequences=return_seq,
                dropout=dropout,
                name=f"rnn_layer_{i}",
            )
        x = rnn_layer(x)

    probs = layers.Dense(vocab_size, activation="softmax", name="output_dense")(x)

    model = Model(
        inputs=[cnn_input, word_input],
        outputs=probs,
        name=f"{rnn_type}_captioner_pre_inject",
    )
    return model


def build_rnn_decoder_init_inject(
    vocab_size: int,
    embed_dim: int = 256,
    rnn_units: int = 256,
    feature_dim: int = 2048,
    num_rnn_layers: int = 1,
    max_caption_len: int = 30,
    rnn_type: str = "lstm",
    dropout: float = 0.0,
) -> Model:
    rnn_type = rnn_type.lower()

    cnn_input = keras.Input(shape=(feature_dim,), name="cnn_input")
    word_input = keras.Input(shape=(None,), dtype="int32", name="word_input")

    h0 = layers.Dense(rnn_units, activation="tanh", name="h0_projection")(cnn_input)

    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(word_input)

    for i in range(num_rnn_layers):
        is_last = i == num_rnn_layers - 1
        if rnn_type == "lstm":
            c0 = layers.Dense(rnn_units, activation="tanh", name=f"c0_projection_{i}")(
                cnn_input
            )
            rnn_lyr = layers.LSTM(
                rnn_units,
                return_sequences=True,
                dropout=dropout,
                name=f"rnn_layer_{i}",
            )
            x = rnn_lyr(x, initial_state=[h0, c0])
        else:
            rnn_lyr = layers.SimpleRNN(
                rnn_units,
                return_sequences=True,
                dropout=dropout,
                name=f"rnn_layer_{i}",
            )
            x = rnn_lyr(x, initial_state=h0)
        if i < num_rnn_layers - 1:
            h0 = layers.Dense(
                rnn_units, activation="tanh", name=f"h0_projection_{i+1}"
            )(cnn_input)

    probs = layers.Dense(vocab_size, activation="softmax", name="output_dense")(x)

    model = Model(
        inputs=[cnn_input, word_input],
        outputs=probs,
        name=f"{rnn_type}_captioner_init_inject",
    )
    return model

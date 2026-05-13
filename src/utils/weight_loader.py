import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.rnn.layers import Embedding, SimpleRNNCell, LSTMCell, DenseProjection, DenseOutput


def load_scratch_components(model_path: str, decoder_type: str) -> dict:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)

    proj_layer = model.get_layer("image_projection")
    cnn_projection = DenseProjection.from_keras(proj_layer)

    emb_layer = model.get_layer("embedding")
    embedding = Embedding.from_keras(emb_layer)

    rnn_cells = []
    idx = 0
    while True:
        try:
            rnn_layer = model.get_layer(f"rnn_layer_{idx}")
            if decoder_type.lower() == "lstm":
                rnn_cells.append(LSTMCell.from_keras(rnn_layer))
            else:
                rnn_cells.append(SimpleRNNCell.from_keras(rnn_layer))
            idx += 1
        except ValueError:
            break

    if not rnn_cells:
        raise ValueError(
            f"No layers named 'rnn_layer_0' found in model at {model_path}. "
            "Check that the model was built with build_rnn_decoder_pre_inject()."
        )

    out_layer = model.get_layer("output_dense")
    output_dense = DenseOutput.from_keras(out_layer)

    return {
        "cnn_projection": cnn_projection,
        "embedding": embedding,
        "rnn_cells": rnn_cells,
        "output_dense": output_dense,
        "num_layers": len(rnn_cells),
        "hidden_size": rnn_cells[0].units,
        "embed_dim": embedding.embed_dim,
        "vocab_size": embedding.vocab_size,
    }

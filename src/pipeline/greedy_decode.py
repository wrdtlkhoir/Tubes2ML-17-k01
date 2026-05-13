import numpy as np


def greedy_decode_rnn(
    feature: np.ndarray,
    cnn_projection,
    embedding_layer,
    rnn_cells: list,
    output_dense,
    vocab: dict,
    idx2word: dict,
    max_len: int = 20,
) -> str:
    x = cnn_projection.forward(feature[np.newaxis, :])

    hidden = [np.zeros((1, cell.units)) for cell in rnn_cells]

    h_in = x
    for i, cell in enumerate(rnn_cells):
        h_new = cell.step(h_in, hidden[i])
        hidden[i] = h_new
        h_in = h_new

    token = vocab["<start>"]
    caption = []

    for _ in range(max_len):
        emb = embedding_layer.forward(np.array([token]))  # (1, embed_dim)

        h_in = emb
        for i, cell in enumerate(rnn_cells):
            h_new = cell.step(h_in, hidden[i])
            hidden[i] = h_new
            h_in = h_new

        logits = output_dense.forward(h_in)  # (1, vocab_size)
        token = int(np.argmax(logits[0]))
        word = idx2word.get(token, "<unk>")

        if word == "<end>":
            break
        caption.append(word)

    return " ".join(caption)


def greedy_decode_lstm(
    feature: np.ndarray,
    cnn_projection,
    embedding_layer,
    lstm_cells: list,
    output_dense,
    vocab: dict,
    idx2word: dict,
    max_len: int = 20,
) -> str:
    x = cnn_projection.forward(feature[np.newaxis, :])

    h_states = [np.zeros((1, cell.units)) for cell in lstm_cells]
    c_states = [np.zeros((1, cell.units)) for cell in lstm_cells]

    h_in = x
    for i, cell in enumerate(lstm_cells):
        h_new, c_new = cell.step(h_in, h_states[i], c_states[i])
        h_states[i], c_states[i] = h_new, c_new
        h_in = h_new

    # Greedy decode
    token = vocab["<start>"]
    caption = []

    for _ in range(max_len):
        emb = embedding_layer.forward(np.array([token]))  # (1, embed_dim)

        h_in = emb
        for i, cell in enumerate(lstm_cells):
            h_new, c_new = cell.step(h_in, h_states[i], c_states[i])
            h_states[i], c_states[i] = h_new, c_new
            h_in = h_new

        logits = output_dense.forward(h_in)  # (1, vocab_size)
        token = int(np.argmax(logits[0]))
        word = idx2word.get(token, "<unk>")

        if word == "<end>":
            break
        caption.append(word)

    return " ".join(caption)

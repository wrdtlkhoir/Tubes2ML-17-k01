import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import time

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rnn.layers import (
    Embedding,
    SimpleRNNCell,
    LSTMCell,
    DenseProjection,
    DenseOutput,
)


class CaptioningPipeline:
    def __init__(
        self,
        embedding: Embedding,
        projection: DenseProjection,
        output_layer: DenseOutput,
        vocab: Dict[str, int],
        max_caption_len: int = 30,
    ):
        self.embedding = embedding
        self.projection = projection
        self.output_layer = output_layer
        self.vocab = vocab
        self.id2word = {v: k for k, v in vocab.items()}
        self.start_id = vocab.get("<start>", vocab.get("startseq", 1))
        self.end_id = vocab.get("<end>", vocab.get("endseq", 2))
        self.pad_id = vocab.get("<pad>", vocab.get("", 0))
        self.max_len = max_caption_len

    def _rnn_init(self, batch_size: int) -> tuple:
        raise NotImplementedError

    def _rnn_step(self, x_t: np.ndarray, state: tuple) -> Tuple[np.ndarray, tuple]:
        raise NotImplementedError

    def _prepare_image_token(self, cnn_features: np.ndarray) -> np.ndarray:
        return self.projection.forward(cnn_features)  # (N, embed_dim)

    def greedy_decode(
        self, cnn_features: np.ndarray, max_len: Optional[int] = None
    ) -> List[List[str]]:
        max_len = max_len or self.max_len
        N = cnn_features.shape[0]
        state = self._rnn_init(N)

        x_t = self._prepare_image_token(cnn_features)  # (N, embed_dim)
        _, state = self._rnn_step(x_t, state)

        tokens = np.full(N, self.start_id, dtype=np.int32)
        done = np.zeros(N, dtype=bool)
        words = [[] for _ in range(N)]

        for _ in range(max_len):
            x_t = self.embedding.forward(tokens)  # (N, embed_dim)
            h_t, state = self._rnn_step(x_t, state)
            logits = self.output_layer.forward(h_t)  # (N, vocab_size)
            tokens = np.argmax(logits, axis=-1)  # (N,)

            for i in range(N):
                if not done[i]:
                    if tokens[i] == self.end_id:
                        done[i] = True
                    else:
                        words[i].append(self.id2word.get(int(tokens[i]), "<unk>"))
            if done.all():
                break

        return words

    def beam_search_decode(
        self,
        cnn_feature: np.ndarray,
        beam_width: int = 5,
        max_len: Optional[int] = None,
    ) -> List[str]:
        max_len = max_len or self.max_len
        feat = cnn_feature[np.newaxis, :]

        state = self._rnn_init(1)
        x_img = self._prepare_image_token(feat)
        _, state = self._rnn_step(x_img, state)

        start_emb = self.embedding.forward(np.array([self.start_id]))
        h_t, state0 = self._rnn_step(start_emb, state)
        logits = self.output_layer.forward(h_t)
        log_probs = np.log(logits[0] + 1e-12)

        top_ids = np.argsort(log_probs)[-beam_width:][::-1]
        beams = [
            (
                log_probs[idx],
                [self.id2word.get(int(idx), "<unk>")],
                self._clone_state(state0),
            )
            for idx in top_ids
            if idx != self.end_id
        ]
        completed = []

        for _ in range(max_len - 1):
            candidates = []
            for score, words_so_far, beam_state in beams:
                last_word = words_so_far[-1]
                last_id = self.vocab.get(last_word, self.pad_id)
                emb = self.embedding.forward(np.array([last_id]))
                h_t, new_state = self._rnn_step(emb, beam_state)
                logits = self.output_layer.forward(h_t)
                lp = np.log(logits[0] + 1e-12)
                top_ids = np.argsort(lp)[-beam_width:][::-1]

                for idx in top_ids:
                    new_score = score + lp[idx]
                    if idx == self.end_id:
                        completed.append((new_score, words_so_far))
                    else:
                        candidates.append(
                            (
                                new_score,
                                words_so_far + [self.id2word.get(int(idx), "<unk>")],
                                self._clone_state(new_state),
                            )
                        )

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        if not completed:
            completed = [(s, w) for s, w, _ in beams]

        completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        return completed[0][1]

    def _clone_state(self, state: tuple) -> tuple:
        return tuple(s.copy() for s in state)

    def timed_greedy_decode(
        self, cnn_features: np.ndarray, max_len: Optional[int] = None
    ) -> Tuple[List[List[str]], float]:
        t0 = time.perf_counter()
        caps = self.greedy_decode(cnn_features, max_len)
        return caps, time.perf_counter() - t0


class SimpleRNNCaptioner(CaptioningPipeline):
    def __init__(
        self,
        embedding,
        projection,
        rnn_cells: List[SimpleRNNCell],
        output_layer,
        vocab,
        max_caption_len=30,
    ):
        super().__init__(embedding, projection, output_layer, vocab, max_caption_len)
        self.rnn_cells = rnn_cells

    def _rnn_init(self, batch_size: int) -> tuple:
        """State = tuple of h for each layer."""
        return tuple(np.zeros((batch_size, cell.units)) for cell in self.rnn_cells)

    def _rnn_step(self, x_t: np.ndarray, state: tuple) -> Tuple[np.ndarray, tuple]:
        new_state = []
        h = x_t
        for i, cell in enumerate(self.rnn_cells):
            h = cell.step(h, state[i])
            new_state.append(h)
        return h, tuple(new_state)

    @classmethod
    def from_keras(
        cls,
        keras_model,
        vocab: Dict[str, int],
        max_caption_len: int = 30,
        rnn_layer_names: Optional[List[str]] = None,
        projection_layer_name: str = "image_projection",
        output_layer_name: str = "output_dense",
        embedding_layer_name: str = "embedding",
    ) -> "SimpleRNNCaptioner":
        embedding = Embedding.from_keras(keras_model.get_layer(embedding_layer_name))
        projection = DenseProjection.from_keras(
            keras_model.get_layer(projection_layer_name)
        )
        output_lyr = DenseOutput.from_keras(keras_model.get_layer(output_layer_name))

        if rnn_layer_names is None:
            rnn_layer_names = [
                l.name for l in keras_model.layers if l.name.startswith("rnn_layer_")
            ]
            if not rnn_layer_names:
                rnn_layer_names = [
                    l.name
                    for l in keras_model.layers
                    if l.__class__.__name__ == "SimpleRNN"
                ]

        cells = [
            SimpleRNNCell.from_keras(keras_model.get_layer(n)) for n in rnn_layer_names
        ]

        return cls(embedding, projection, cells, output_lyr, vocab, max_caption_len)


class LSTMCaptioner(CaptioningPipeline):
    def __init__(
        self,
        embedding,
        projection,
        lstm_cells: List[LSTMCell],
        output_layer,
        vocab,
        max_caption_len=30,
    ):
        super().__init__(embedding, projection, output_layer, vocab, max_caption_len)
        self.lstm_cells = lstm_cells

    def _rnn_init(self, batch_size: int) -> tuple:
        state = []
        for cell in self.lstm_cells:
            state.append(np.zeros((batch_size, cell.units)))  # h
            state.append(np.zeros((batch_size, cell.units)))  # c
        return tuple(state)

    def _rnn_step(self, x_t: np.ndarray, state: tuple) -> Tuple[np.ndarray, tuple]:
        new_state = []
        h = x_t
        for i, cell in enumerate(self.lstm_cells):
            h_prev, c_prev = state[2 * i], state[2 * i + 1]
            h, c = cell.step(h, h_prev, c_prev)
            new_state.extend([h, c])
        return h, tuple(new_state)

    @classmethod
    def from_keras(
        cls,
        keras_model,
        vocab: Dict[str, int],
        max_caption_len: int = 30,
        lstm_layer_names: Optional[List[str]] = None,
        projection_layer_name: str = "image_projection",
        output_layer_name: str = "output_dense",
        embedding_layer_name: str = "embedding",
    ) -> "LSTMCaptioner":
        embedding = Embedding.from_keras(keras_model.get_layer(embedding_layer_name))
        projection = DenseProjection.from_keras(
            keras_model.get_layer(projection_layer_name)
        )
        output_lyr = DenseOutput.from_keras(keras_model.get_layer(output_layer_name))

        if lstm_layer_names is None:
            lstm_layer_names = [
                l.name for l in keras_model.layers if l.name.startswith("rnn_layer_")
            ]
            if not lstm_layer_names:
                lstm_layer_names = [
                    l.name for l in keras_model.layers if l.__class__.__name__ == "LSTM"
                ]

        cells = [
            LSTMCell.from_keras(keras_model.get_layer(n)) for n in lstm_layer_names
        ]

        return cls(embedding, projection, cells, output_lyr, vocab, max_caption_len)

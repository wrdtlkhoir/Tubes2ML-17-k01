import numpy as np
from typing import Optional, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn.layers import Dense, Softmax, ReLU


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Embedding:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings  # (vocab_size, embed_dim)
        self.vocab_size, self.embed_dim = embeddings.shape

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embeddings[token_ids]

    def backward(self, d_out: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        d_embeddings = np.zeros_like(self.embeddings)
        np.add.at(d_embeddings, token_ids, d_out)
        return d_embeddings

    @classmethod
    def from_keras(cls, keras_layer) -> "Embedding":
        weights = keras_layer.get_weights()  # [embedding_matrix]
        return cls(weights[0])


class SimpleRNNCell:
    def __init__(self, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray):
        self.W_xh = W_xh
        self.W_hh = W_hh
        self.b_h = b_h
        self.units = W_hh.shape[0]

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        h_t = _tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        return h_t

    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
        return_sequences: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N, T, _ = x.shape
        h = np.zeros((N, self.units)) if h0 is None else h0.copy()

        hs = np.zeros((N, T, self.units))
        for t in range(T):
            h = self.step(x[:, t, :], h)
            hs[:, t, :] = h

        output = hs if return_sequences else h
        return output, h

    def backward_step(
        self, d_h: np.ndarray, x_t: np.ndarray, h_prev: np.ndarray, h_t: np.ndarray
    ):
        d_pre = d_h * (1 - h_t**2)

        d_W_xh = x_t.T @ d_pre
        d_W_hh = h_prev.T @ d_pre
        d_b_h = d_pre.sum(axis=0)
        d_x_t = d_pre @ self.W_xh.T
        d_h_prev = d_pre @ self.W_hh.T

        grads = {"W_xh": d_W_xh, "W_hh": d_W_hh, "b_h": d_b_h}
        return d_x_t, d_h_prev, grads

    def backward(
        self, d_out: np.ndarray, xs: np.ndarray, hs: np.ndarray, h0: np.ndarray
    ):
        N, T, _ = xs.shape
        d_xs = np.zeros_like(xs)
        d_h = np.zeros((N, self.units))

        grads = {
            "W_xh": np.zeros_like(self.W_xh),
            "W_hh": np.zeros_like(self.W_hh),
            "b_h": np.zeros_like(self.b_h),
        }

        if d_out.ndim == 3:
            d_out_seq = d_out
        else:
            d_out_seq = np.zeros((N, T, self.units))
            d_out_seq[:, -1, :] = d_out

        for t in reversed(range(T)):
            d_h_total = d_h + d_out_seq[:, t, :]
            h_prev = hs[:, t - 1, :] if t > 0 else h0
            d_x_t, d_h, step_grads = self.backward_step(
                d_h_total, xs[:, t, :], h_prev, hs[:, t, :]
            )
            d_xs[:, t, :] = d_x_t
            for k in grads:
                grads[k] += step_grads[k]

        return d_xs, grads

    @classmethod
    def from_keras(cls, keras_layer) -> "SimpleRNNCell":
        W_xh, W_hh, b_h = keras_layer.get_weights()
        return cls(W_xh, W_hh, b_h)


class LSTMCell:
    def __init__(self, W_x: np.ndarray, W_h: np.ndarray, b: np.ndarray):
        self.W_x = W_x
        self.W_h = W_h
        self.b = b  # shape (4*units,)
        self.units = W_h.shape[0]

    def step(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b  # (N, 4*units)
        u = self.units

        i_gate = _sigmoid(gates[:, 0 * u : 1 * u])
        f_gate = _sigmoid(gates[:, 1 * u : 2 * u])
        g_gate = _tanh(gates[:, 2 * u : 3 * u])
        o_gate = _sigmoid(gates[:, 3 * u : 4 * u])

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * _tanh(c_t)

        return h_t, c_t

    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
        c0: Optional[np.ndarray] = None,
        return_sequences: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, T, _ = x.shape
        h = np.zeros((N, self.units)) if h0 is None else h0.copy()
        c = np.zeros((N, self.units)) if c0 is None else c0.copy()

        hs = np.zeros((N, T, self.units))
        cs = np.zeros((N, T, self.units))

        for t in range(T):
            h, c = self.step(x[:, t, :], h, c)
            hs[:, t, :] = h
            cs[:, t, :] = c

        output = hs if return_sequences else h
        return output, h, c

    def backward(
        self,
        d_out: np.ndarray,
        xs: np.ndarray,
        hs: np.ndarray,
        cs: np.ndarray,
        h0: np.ndarray,
        c0: np.ndarray,
    ):
        N, T, _ = xs.shape
        u = self.units
        d_xs = np.zeros_like(xs)
        d_h = np.zeros((N, u))
        d_c = np.zeros((N, u))

        grads = {
            "W_x": np.zeros_like(self.W_x),
            "W_h": np.zeros_like(self.W_h),
            "b": np.zeros_like(self.b),
        }

        if d_out.ndim == 3:
            d_out_seq = d_out
        else:
            d_out_seq = np.zeros((N, T, u))
            d_out_seq[:, -1, :] = d_out

        for t in reversed(range(T)):
            h_prev = hs[:, t - 1, :] if t > 0 else h0
            c_prev = cs[:, t - 1, :] if t > 0 else c0

            d_h_total = d_h + d_out_seq[:, t, :]

            # Recompute gate values
            gates = xs[:, t, :] @ self.W_x + h_prev @ self.W_h + self.b
            i_gate = _sigmoid(gates[:, 0 * u : 1 * u])
            f_gate = _sigmoid(gates[:, 1 * u : 2 * u])
            g_gate = _tanh(gates[:, 2 * u : 3 * u])
            o_gate = _sigmoid(gates[:, 3 * u : 4 * u])
            c_t = cs[:, t, :]
            tanh_ct = _tanh(c_t)

            # d_c
            d_c_total = d_h_total * o_gate * (1 - tanh_ct**2) + d_c

            d_o = d_h_total * tanh_ct * o_gate * (1 - o_gate)
            d_i = d_c_total * g_gate * i_gate * (1 - i_gate)
            d_f = d_c_total * c_prev * f_gate * (1 - f_gate)
            d_g = d_c_total * i_gate * (1 - g_gate**2)

            d_gates = np.concatenate([d_i, d_f, d_g, d_o], axis=1)

            grads["W_x"] += xs[:, t, :].T @ d_gates
            grads["W_h"] += h_prev.T @ d_gates
            grads["b"] += d_gates.sum(axis=0)

            d_xs[:, t, :] = d_gates @ self.W_x.T
            d_h = d_gates @ self.W_h.T
            d_c = d_c_total * f_gate

        return d_xs, grads

    @classmethod
    def from_keras(cls, keras_layer) -> "LSTMCell":
        weights = keras_layer.get_weights()
        W_x = weights[0]  # (input_dim, 4*units)
        W_h = weights[1]  # (units, 4*units)
        if len(weights) == 3:
            b = weights[2]  # (4*units,)
        else:
            b = weights[2] + weights[3]  # sum input & recurrent bias
        return cls(W_x, W_h, b)


class DenseProjection(Dense):
    @classmethod
    def from_keras(
        cls, keras_layer, activation: Optional[str] = None
    ) -> "DenseProjection":
        weight, bias = keras_layer.get_weights()
        return cls(weight, bias, activation)


class DenseOutput(Dense):
    @classmethod
    def from_keras(cls, keras_layer) -> "DenseOutput":
        weight, bias = keras_layer.get_weights()
        return cls(weight, bias, activation="softmax")

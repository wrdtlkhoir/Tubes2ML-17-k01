import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from typing import Dict, List, Optional, Tuple

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rnn.models import build_rnn_decoder_pre_inject, build_rnn_decoder_init_inject
from rnn.preprocessing import (
    build_vocab,
    load_vocab,
    save_vocab,
    build_dataset,
    load_flickr8k_captions,
    PAD_ID,
)

def load_features(features_dir: str, image_names: List[str]) -> Dict[str, np.ndarray]:

    def _reduce_feature(feat: np.ndarray) -> np.ndarray:
        if feat.ndim == 1 and feat.size % 2048 == 0 and feat.size > 2048:
            feat = feat.reshape(-1, 2048).mean(axis=0)
        elif feat.ndim == 2 and feat.shape[-1] == 2048 and feat.shape[0] > 1:
            feat = feat.mean(axis=0)
        elif feat.ndim == 3 and feat.shape[-1] == 2048:
            feat = feat.mean(axis=(0, 1))
        return feat.astype(np.float32, copy=False)

    features: Dict[str, np.ndarray] = {}
    for name in image_names:
        npy_name = name.replace(".jpg", ".npy").replace(".png", ".npy")
        path = os.path.join(features_dir, npy_name)
        if os.path.exists(path):
            features[name] = _reduce_feature(np.load(path))
    return features


def make_tf_dataset(X_img, X_words, Y, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        ({"cnn_input": X_img, "word_input": X_words}, Y)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def masked_sparse_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


class CaptioningTrainer:
    NUM_LAYERS_VARIANTS = [1, 2, 3]
    HIDDEN_SIZE_VARIANTS = [128, 512]

    def __init__(
        self,
        features_dir: str,
        captions_file: str,
        train_split: List[str],
        val_split: List[str],
        test_split: List[str],
        output_dir: str = "./trained_models/rnn",
        vocab_path: str = "./trained_models/rnn/vocab.json",
        embed_dim: int = 256,
        max_caption_len: int = 30,
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5,
    ):

        self.features_dir = features_dir
        self.captions_file = captions_file
        self.output_dir = output_dir
        self.vocab_path = vocab_path
        self.embed_dim = embed_dim
        self.max_len = max_caption_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        os.makedirs(output_dir, exist_ok=True)

        def _ensure_list(split, name):
            if isinstance(split, (list, tuple, np.ndarray)):
                return list(split)
            raise TypeError(
                f"{name} must be a list of image filenames, got {type(split).__name__}"
            )

        self.train_names = _ensure_list(train_split, "train_split")
        self.val_names = _ensure_list(val_split, "val_split")
        self.test_names = _ensure_list(test_split, "test_split")

        self.all_captions = load_flickr8k_captions(captions_file)

        if os.path.exists(vocab_path):
            self.vocab = load_vocab(vocab_path)
            print(f"[vocab] Loaded {len(self.vocab)} tokens from {vocab_path}")
        else:
            train_caps = [
                c for name in self.train_names for c in self.all_captions.get(name, [])
            ]
            self.vocab = build_vocab(train_caps, min_freq=5)
            save_vocab(self.vocab, vocab_path)
            print(f"[vocab] Built vocab of {len(self.vocab)} tokens → {vocab_path}")

        self.vocab_size = len(self.vocab)
        self.results: Dict = {}

    def _prepare_split(self, names: List[str], include_image_token: bool = True):
        features = load_features(self.features_dir, names)
        X_img, X_words, Y = build_dataset(
            names,
            self.all_captions,
            features,
            self.vocab,
            self.max_len,
            include_image_token=include_image_token,
        )
        feat_dim = X_img.shape[1] if len(X_img) > 0 else 2048
        return X_img, X_words, Y, feat_dim

    def _train_one(self, model: keras.Model, model_name: str, train_ds, val_ds) -> dict:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=masked_sparse_crossentropy,
        )
        ckpt_path = os.path.join(self.output_dir, f"{model_name}.h5")
        cb = [
            callbacks.ModelCheckpoint(
                ckpt_path,
                save_best_only=True,
                monitor="val_loss",
                save_weights_only=False,
            ),
            callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]
        t0 = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=cb,
            verbose=1,
        )
        elapsed = time.perf_counter() - t0

        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return {"history": hist, "train_time_s": elapsed, "model_path": ckpt_path}

    def run(
        self,
        injection: str = "pre",
        rnn_types: Tuple[str, ...] = ("rnn", "lstm"),
    ):
        injection = injection.strip().lower()
        if injection not in {"pre", "init"}:
            raise ValueError(
                f"injection must be 'pre' or 'init', got {injection!r}"
            )
        include_image_token = injection == "pre"

        print("[data] Preparing training split …")
        X_img_tr, X_w_tr, Y_tr, feat_dim = self._prepare_split(
            self.train_names, include_image_token=include_image_token
        )
        print("[data] Preparing validation split …")
        X_img_vl, X_w_vl, Y_vl, _ = self._prepare_split(
            self.val_names, include_image_token=include_image_token
        )
        if include_image_token:
            expected_tr = X_w_tr.shape[1] + 1
            expected_vl = X_w_vl.shape[1] + 1
            if Y_tr.shape[1] != expected_tr:
                raise ValueError(
                    f"pre-inject expects train Y length {expected_tr}, got {Y_tr.shape[1]}"
                )
            if Y_vl.shape[1] != expected_vl:
                raise ValueError(
                    f"pre-inject expects val Y length {expected_vl}, got {Y_vl.shape[1]}"
                )
        else:
            expected_tr = X_w_tr.shape[1]
            expected_vl = X_w_vl.shape[1]
            if Y_tr.shape[1] != expected_tr:
                Y_tr = Y_tr[:, :expected_tr]
            if Y_vl.shape[1] != expected_vl:
                Y_vl = Y_vl[:, :expected_vl]

        self.feat_dim = feat_dim

        train_ds = make_tf_dataset(
            X_img_tr, X_w_tr, Y_tr, self.batch_size, shuffle=True
        )
        val_ds = make_tf_dataset(X_img_vl, X_w_vl, Y_vl, self.batch_size, shuffle=False)

        for rnn_type in rnn_types:
            for num_layers in self.NUM_LAYERS_VARIANTS:
                for hidden_size in self.HIDDEN_SIZE_VARIANTS:
                    name = (
                        f"{rnn_type}_{injection}inject" f"_L{num_layers}_H{hidden_size}"
                    )
                    print(f"\n=== Training {name} ===")

                    if injection == "pre":
                        model = build_rnn_decoder_pre_inject(
                            vocab_size=self.vocab_size,
                            embed_dim=self.embed_dim,
                            rnn_units=hidden_size,
                            feature_dim=feat_dim,
                            num_rnn_layers=num_layers,
                            max_caption_len=self.max_len,
                            rnn_type=rnn_type,
                        )
                    else:
                        model = build_rnn_decoder_init_inject(
                            vocab_size=self.vocab_size,
                            embed_dim=self.embed_dim,
                            rnn_units=hidden_size,
                            feature_dim=feat_dim,
                            num_rnn_layers=num_layers,
                            max_caption_len=self.max_len,
                            rnn_type=rnn_type,
                        )

                    result = self._train_one(model, name, train_ds, val_ds)
                    result.update(
                        {
                            "rnn_type": rnn_type,
                            "injection": injection,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "vocab_size": self.vocab_size,
                            "embed_dim": self.embed_dim,
                            "feat_dim": feat_dim,
                            "max_caption_len": self.max_len,
                        }
                    )
                    self.results[name] = result
                    print(
                        f"  Done. Best val_loss="
                        f"{min(result['history']['val_loss']):.4f}"
                    )

        self._save_results()

    def _save_results(self):
        path = os.path.join(self.output_dir, "training_results_rnn.json")
        serialisable = {}
        for k, v in self.results.items():
            serialisable[k] = {
                kk: vv for kk, vv in v.items() if kk != "model"
            }  # drop non-serialisable
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\n[results] Saved to {path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--captions_file", required=True)
    ap.add_argument("--train_split", required=True)
    ap.add_argument("--val_split", required=True)
    ap.add_argument("--test_split", required=True)
    ap.add_argument("--output_dir", default="./trained_models/rnn")
    ap.add_argument("--vocab_path", default="./trained_models/rnn/vocab.json")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--injection", default="pre", choices=["pre", "init"])
    args = ap.parse_args()

    trainer = CaptioningTrainer(
        features_dir=args.features_dir,
        captions_file=args.captions_file,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        embed_dim=args.embed_dim,
        max_caption_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    trainer.run(injection=args.injection)

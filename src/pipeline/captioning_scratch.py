import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.weight_loader import load_scratch_components
from src.pipeline.greedy_decode import greedy_decode_rnn, greedy_decode_lstm

_INCEPTION_SIZE = (299, 299)
_VGG_SIZE = (224, 224)


class ImageCaptionerScratch:

    def __init__(self, decoder_type: str = "lstm"):
        assert decoder_type in ("rnn", "lstm"), "decoder_type must be 'rnn' or 'lstm'"
        self.decoder_type = decoder_type
        self._cnn_encoder = None
        self._target_size = _INCEPTION_SIZE
        self._components = None
        self._vocab = None
        self._idx2word = None

    def load_weights(
        self,
        cnn_encoder_path: str,
        decoder_weights_path: str,
        vocab_path: str = "data/captions/vocab.json",
    ):
        import tensorflow as tf

        # CNN encoder
        name = cnn_encoder_path.strip()
        if name == "InceptionV3":
            self._cnn_encoder = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", pooling="avg",
                input_shape=(*_INCEPTION_SIZE, 3),
            )
            self._target_size = _INCEPTION_SIZE
        elif name == "VGG16":
            self._cnn_encoder = tf.keras.applications.VGG16(
                include_top=False, weights="imagenet", pooling="avg",
                input_shape=(*_VGG_SIZE, 3),
            )
            self._target_size = _VGG_SIZE
        else:
            self._cnn_encoder = tf.keras.models.load_model(name, compile=False)
            try:
                inp_shape = self._cnn_encoder.input_shape
                self._target_size = (inp_shape[1], inp_shape[2])
            except Exception:
                self._target_size = _INCEPTION_SIZE
        self._cnn_encoder.trainable = False

        self._components = load_scratch_components(decoder_weights_path, self.decoder_type)

        import json
        with open(vocab_path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._idx2word = {v: k for k, v in self._vocab.items()}

    def _extract_feature(self, image_path: str) -> np.ndarray:
        from PIL import Image

        img = Image.open(image_path).convert("RGB").resize(self._target_size)
        arr = np.array(img, dtype=np.float32)[np.newaxis, :] / 255.0
        return self._cnn_encoder.predict(arr, verbose=0)[0]  # (feature_dim,)

    def _caption_from_feature(self, feature: np.ndarray, max_len: int) -> str:
        c = self._components
        fn = greedy_decode_lstm if self.decoder_type == "lstm" else greedy_decode_rnn
        return fn(
            feature,
            c["cnn_projection"],
            c["embedding"],
            c["rnn_cells"],
            c["output_dense"],
            self._vocab,
            self._idx2word,
            max_len=max_len,
        )

    def predict(self, image_path: str, max_len: int = 20) -> str:
        if self._components is None:
            raise RuntimeError("Call load_weights() first.")
        feature = self._extract_feature(image_path)
        return self._caption_from_feature(feature, max_len)

    def predict_from_feature(self, feature: np.ndarray, max_len: int = 20) -> str:
        if self._components is None:
            raise RuntimeError("Call load_weights() first.")
        return self._caption_from_feature(feature, max_len)

    def predict_batch(self, image_paths: list, max_len: int = 20) -> list:
        return [self.predict(p, max_len) for p in image_paths]

    def predict_batch_from_features(
        self, features: np.ndarray, max_len: int = 20
    ) -> list:
        return [self._caption_from_feature(features[i], max_len) for i in range(len(features))]

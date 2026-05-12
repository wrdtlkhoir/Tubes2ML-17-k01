import os
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rnn.preprocessing import (
    load_vocab,
    get_id2word,
    load_flickr8k_captions,
    encode_caption,
    PAD_ID,
    START_ID,
    END_ID,
)
from rnn.forward_propagation import SimpleRNNCaptioner, LSTMCaptioner


def keras_greedy_decode(
    keras_model, cnn_features: np.ndarray, vocab: Dict[str, int], max_len: int = 30
) -> List[List[str]]:
    id2word = get_id2word(vocab)
    N = cnn_features.shape[0]
    end_id = vocab.get("<end>", 2)
    pad_id = vocab.get("<pad>", 0)

    sequences = np.full((N, 1), vocab.get("<start>", 1), dtype=np.int32)
    done = np.zeros(N, dtype=bool)
    words_out = [[] for _ in range(N)]

    for _ in range(max_len):
        preds = keras_model.predict(
            {"cnn_input": cnn_features, "word_input": sequences}, verbose=0
        )
        next_token = np.argmax(preds[:, -1, :], axis=-1)

        for i in range(N):
            if not done[i]:
                if next_token[i] == end_id:
                    done[i] = True
                else:
                    words_out[i].append(id2word.get(int(next_token[i]), "<unk>"))

        if done.all():
            break
        sequences = np.concatenate([sequences, next_token[:, np.newaxis]], axis=1)

    return words_out


def compute_bleu4(
    hypotheses: List[List[str]], references: List[List[List[str]]]
) -> float:
    smoother = SmoothingFunction().method4
    return corpus_bleu(references, hypotheses, smoothing_function=smoother)


def compute_meteor(
    hypotheses: List[List[str]], references: List[List[List[str]]]
) -> float:
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp if isinstance(hyp, (list, tuple)) else str(hyp).split()
        ref_tokens = [
            r if isinstance(r, (list, tuple)) else str(r).split() for r in refs
        ]
        scores.append(meteor_score(ref_tokens, hyp_tokens))
    return float(np.mean(scores)) if scores else 0.0


def get_references(
    image_names: List[str], all_captions: Dict[str, List[str]]
) -> List[List[List[str]]]:
    references = []
    for name in image_names:
        caps = all_captions.get(name, [])
        refs = [c.lower().split() for c in caps]
        references.append(refs)
    return references


class CaptioningEvaluator:
    def __init__(
        self,
        models_dir: str,
        features_dir: str,
        vocab_path: str,
        captions_file: str,
        test_split: List[str],
    ):
        self.models_dir = models_dir
        self.features_dir = features_dir
        self.vocab = load_vocab(vocab_path)
        self.id2word = get_id2word(self.vocab)
        self.all_captions = load_flickr8k_captions(captions_file)
        if isinstance(test_split, (list, tuple, np.ndarray)):
            self.test_names = list(test_split)
        else:
            raise TypeError(
                f"test_split must be a list of image filenames, got {type(test_split).__name__}"
            )
        self.results: Dict = {}

        self._test_features, self._test_img_names = self._load_features()
        self._references = get_references(self._test_img_names, self.all_captions)

    def _load_features(self):
        def _reduce_feature(feat: np.ndarray) -> np.ndarray:
            if feat.ndim == 1 and feat.size % 2048 == 0 and feat.size > 2048:
                feat = feat.reshape(-1, 2048).mean(axis=0)
            elif feat.ndim == 2 and feat.shape[-1] == 2048 and feat.shape[0] > 1:
                feat = feat.mean(axis=0)
            elif feat.ndim == 3 and feat.shape[-1] == 2048:
                feat = feat.mean(axis=(0, 1))
            return feat.astype(np.float32, copy=False)

        imgs, names = [], []
        for name in self.test_names:
            npy = name.replace(".jpg", ".npy").replace(".png", ".npy")
            path = os.path.join(self.features_dir, npy)
            if os.path.exists(path):
                imgs.append(_reduce_feature(np.load(path)))
                names.append(name)
        return (
            np.array(imgs, dtype=np.float32)
            if imgs
            else np.empty((0,), dtype=np.float32)
        ), names

    def evaluate_keras(self, model_name: str, max_len: int = 30) -> Dict:
        import tensorflow as tf
        from tensorflow import keras

        path = os.path.join(self.models_dir, f"{model_name}.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        model = keras.models.load_model(
            path,
            compile=False,
            custom_objects={"masked_sparse_crossentropy": lambda y, p: tf.zeros(())},
        )

        t0 = time.perf_counter()
        hyps = keras_greedy_decode(model, self._test_features, self.vocab, max_len)
        elapsed = time.perf_counter() - t0

        bleu4 = compute_bleu4(hyps, self._references)
        meteor = compute_meteor(hyps, self._references)

        result = {
            "bleu4": bleu4,
            "meteor": meteor,
            "decode_time_s": elapsed,
            "source": "keras",
            "model_name": model_name,
        }
        self.results[f"{model_name}_keras"] = result
        return result

    def evaluate_scratch(
        self,
        model_name: str,
        rnn_type: str,
        max_len: int = 30,
        rnn_layer_names: Optional[List[str]] = None,
        lstm_layer_names: Optional[List[str]] = None,
        projection_layer_name: str = "image_projection",
        output_layer_name: str = "output_dense",
        embedding_layer_name: str = "embedding",
    ) -> Dict:
        import tensorflow as tf
        from tensorflow import keras

        path = os.path.join(self.models_dir, f"{model_name}.h5")
        keras_model = keras.models.load_model(
            path,
            compile=False,
            custom_objects={"masked_sparse_crossentropy": lambda y, p: tf.zeros(())},
        )

        if rnn_type == "lstm":
            captioner = LSTMCaptioner.from_keras(
                keras_model,
                self.vocab,
                max_len,
                lstm_layer_names=lstm_layer_names,
                projection_layer_name=projection_layer_name,
                output_layer_name=output_layer_name,
                embedding_layer_name=embedding_layer_name,
            )
        else:
            captioner = SimpleRNNCaptioner.from_keras(
                keras_model,
                self.vocab,
                max_len,
                rnn_layer_names=rnn_layer_names,
                projection_layer_name=projection_layer_name,
                output_layer_name=output_layer_name,
                embedding_layer_name=embedding_layer_name,
            )

        t0 = time.perf_counter()
        hyps = captioner.greedy_decode(self._test_features, max_len)
        elapsed = time.perf_counter() - t0

        bleu4 = compute_bleu4(hyps, self._references)
        meteor = compute_meteor(hyps, self._references)

        result = {
            "bleu4": bleu4,
            "meteor": meteor,
            "decode_time_s": elapsed,
            "source": "scratch",
            "model_name": model_name,
        }
        self.results[f"{model_name}_scratch"] = result
        return result

    def evaluate_max_len_variation(
        self, model_name: str, rnn_type: str, max_lens: List[int] = [10, 20, 30, 40, 50]
    ) -> Dict[int, Dict]:
        variation_results = {}
        for ml in max_lens:
            print(f"  max_len={ml} …", end=" ", flush=True)
            res = self.evaluate_scratch(model_name, rnn_type, max_len=ml)
            variation_results[ml] = res
            print(f"BLEU-4={res['bleu4']:.4f}")
        self.results[f"{model_name}_maxlen_variation"] = variation_results
        return variation_results

    def qualitative_analysis(
        self,
        model_name_rnn: str,
        model_name_lstm: str,
        n_samples: int = 10,
        max_len: int = 30,
    ) -> List[Dict]:

        import tensorflow as tf
        from tensorflow import keras

        def _load_captioner(name, rnn_type):
            path = os.path.join(self.models_dir, f"{name}.h5")
            km = keras.models.load_model(path, compile=False)
            cls = LSTMCaptioner if rnn_type == "lstm" else SimpleRNNCaptioner
            return cls.from_keras(km, self.vocab, max_len)

        captioner_rnn = _load_captioner(model_name_rnn, "rnn")
        captioner_lstm = _load_captioner(model_name_lstm, "lstm")

        all_results = []
        features = self._test_features
        smooth = SmoothingFunction().method4

        for i, (feat, name, refs) in enumerate(
            zip(features, self._test_img_names, self._references)
        ):
            feat_batch = feat[np.newaxis, :]
            rnn_words = captioner_rnn.greedy_decode(feat_batch, max_len)[0]
            lstm_words = captioner_lstm.greedy_decode(feat_batch, max_len)[0]

            rnn_bleu = corpus_bleu([refs], [rnn_words], smoothing_function=smooth)
            lstm_bleu = corpus_bleu([refs], [lstm_words], smoothing_function=smooth)

            all_results.append(
                {
                    "image": name,
                    "rnn_caption": " ".join(rnn_words),
                    "lstm_caption": " ".join(lstm_words),
                    "references": [" ".join(r) for r in refs],
                    "rnn_bleu4": rnn_bleu,
                    "lstm_bleu4": lstm_bleu,
                }
            )

        all_results.sort(key=lambda x: (x["rnn_bleu4"] + x["lstm_bleu4"]) / 2)
        n_each = n_samples // 3
        sampled = (
            all_results[:n_each]
            + all_results[
                len(all_results) // 2
                - n_each // 2 : len(all_results) // 2
                + n_each // 2
            ]
            + all_results[-n_each:]
        )
        return sampled[:n_samples]

    def save_results(self, path: Optional[str] = None):
        path = path or os.path.join(self.models_dir, "eval_results_rnn.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[eval] Results saved to {path}")

    def run_full_evaluation(self, model_configs: List[Dict], max_len: int = 30):
        for cfg in model_configs:
            name = cfg["model_name"]
            rnn_type = cfg["rnn_type"]
            print(f"\n--- {name} ---")

            print("  Keras …")
            self.evaluate_keras(name, max_len)

            print("  From scratch …")
            self.evaluate_scratch(
                name,
                rnn_type,
                max_len,
                rnn_layer_names=cfg.get("rnn_layer_names"),
                lstm_layer_names=cfg.get("lstm_layer_names"),
            )

        self.save_results()

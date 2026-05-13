import time
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def compute_bleu4(references: list, hypotheses: list) -> float:
    refs = [[r.split() for r in ref_list] for ref_list in references]
    hyps = [h.split() for h in hypotheses]
    smoothing = SmoothingFunction().method1
    return corpus_bleu(
        refs, hyps,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )


def compute_sentence_bleu4(references: list, hypothesis: str) -> float:
    ref_tokens = [r.split() for r in references]
    hyp_tokens = hypothesis.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        ref_tokens, hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )


def compute_meteor(references: list, hypotheses: list) -> float:
    scores = []
    for refs, hyp in zip(references, hypotheses):
        score = max(
            meteor_score([r.split() for r in refs], hyp.split())
            for r in refs
        )
        scores.append(score)
    return float(np.mean(scores))


def timed_predict(captioner, image_paths: list, max_len: int = 20):
    start = time.perf_counter()
    captions = [captioner.predict(p, max_len) for p in image_paths]
    elapsed = time.perf_counter() - start
    return captions, elapsed, elapsed / len(image_paths)


def timed_predict_from_features(captioner, features: np.ndarray, max_len: int = 20):
    start = time.perf_counter()
    captions = captioner.predict_batch_from_features(features, max_len)
    elapsed = time.perf_counter() - start
    return captions, elapsed, elapsed / len(features)

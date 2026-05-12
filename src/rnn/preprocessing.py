import re
import json
import os
import csv
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


def clean_caption(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def build_vocab(
    captions: List[str], min_freq: int = 2, max_vocab_size: Optional[int] = None
) -> Dict[str, int]:
    counter: Counter = Counter()
    for cap in captions:
        tokens = clean_caption(cap).split()
        counter.update(tokens)

    words = [w for w, f in counter.items() if f >= min_freq]

    if max_vocab_size is not None:
        words = sorted(words, key=lambda w: -counter[w])
        words = words[: max_vocab_size - len(SPECIAL_TOKENS)]

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word in sorted(words):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_caption(caption: str, vocab: Dict[str, int]) -> List[int]:
    tokens = clean_caption(caption).split()
    ids = [vocab.get(t, UNK_ID) for t in tokens]
    return [START_ID] + ids + [END_ID]


def encode_captions(captions: List[str], vocab: Dict[str, int]) -> List[List[int]]:
    return [encode_caption(c, vocab) for c in captions]


def decode_caption(
    ids: List[int], id2word: Dict[int, str], remove_special: bool = True
) -> str:
    words = [id2word.get(i, UNK_TOKEN) for i in ids]
    if remove_special:
        words = [
            w for w in words if w not in (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)
        ]
    return " ".join(words)


def pad_sequences(
    sequences: List[List[int]],
    max_len: Optional[int] = None,
    padding_value: int = PAD_ID,
    truncating: str = "post",
    padding: str = "post",
) -> np.ndarray:
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    result = np.full((len(sequences), max_len), padding_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if truncating == "post":
            seq = seq[:max_len]
        else:
            seq = seq[-max_len:]

        length = len(seq)
        if padding == "post":
            result[i, :length] = seq
        else:
            result[i, max_len - length :] = seq

    return result


def load_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    captions: Dict[str, List[str]] = {}

    with open(captions_file, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            img_name = row["image"].strip()
            caption = row["caption"].strip()

            captions.setdefault(img_name, []).append(caption)

    return captions


def load_flickr8k_split(split_file: str) -> List[str]:
    with open(split_file, "r") as f:
        return [l.strip() for l in f if l.strip()]


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_id2word(vocab: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in vocab.items()}


def build_dataset(
    image_names: List[str],
    all_captions: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    vocab: Dict[str, int],
    max_len: int,
    include_image_token: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_imgs, X_words_list, Y_list = [], [], []

    target_len = max_len + 1 if include_image_token else max_len

    for img_name in image_names:
        if img_name not in features or img_name not in all_captions:
            continue
        feat = features[img_name]

        for raw_cap in all_captions[img_name]:
            seq = encode_caption(raw_cap, vocab)

            if len(seq) > max_len + 2:
                seq = seq[: max_len + 2]

            x_seq = seq[:-1][:max_len]
            y_seq = seq[1:][:target_len]

            x_padded = x_seq + [PAD_ID] * (max_len - len(x_seq))
            y_padded = y_seq + [PAD_ID] * (target_len - len(y_seq))

            X_imgs.append(feat)
            X_words_list.append(x_padded)
            Y_list.append(y_padded)

    X_img = np.array(X_imgs, dtype=np.float32)
    X_words = np.array(X_words_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)

    return X_img, X_words, Y

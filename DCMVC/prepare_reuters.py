import os
import re
import html
from collections import Counter

import numpy as np
from scipy.io import savemat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


REUTERS_FILE_RE = re.compile(r"reut2-\d{3}\.sgm$")

def extract_split(block: str):
    m = re.search(r'LEWISSPLIT="([^"]+)"', block)
    return m.group(1).upper() if m else ""

def iter_reuters_docs(sgm_text: str):
    # Split on closing tag (fast and good enough)
    parts = sgm_text.split("</REUTERS>")
    for part in parts:
        if "<REUTERS" not in part:
            continue
        yield part + "</REUTERS>"


def extract_topics(block: str):
    m = re.search(r"<TOPICS>(.*?)</TOPICS>", block, flags=re.S)
    if not m:
        return []
    inner = m.group(1)
    return re.findall(r"<D>(.*?)</D>", inner)


def extract_text(block: str):
    title_m = re.search(r"<TITLE>(.*?)</TITLE>", block, flags=re.S)
    body_m = re.search(r"<BODY>(.*?)</BODY>", block, flags=re.S)

    title = title_m.group(1) if title_m else ""
    body = body_m.group(1) if body_m else ""

    txt = f"{title} {body}".strip()
    if not txt:
        m = re.search(r"<TEXT[^>]*>(.*?)</TEXT>", block, flags=re.S)
        txt = m.group(1) if m else ""

    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def load_reuters(data_dir: str, splits=("TRAIN",), single_label_only: bool = True):
    files = [f for f in os.listdir(data_dir) if REUTERS_FILE_RE.search(f)]
    if not files:
        raise FileNotFoundError(f"No reut2-XXX.sgm files in {data_dir}")
    files.sort()

    texts = []
    labels = []

    for f in files:
        path = os.path.join(data_dir, f)
        with open(path, "r", encoding="latin-1") as fp:
            sgm = fp.read()
        for doc in iter_reuters_docs(sgm):
            split_tag = extract_split(doc)
            if splits is not None and split_tag not in splits:
                continue

            topics = extract_topics(doc)
            if not topics:
                continue
            if single_label_only and len(topics) != 1:
                continue

            text = extract_text(doc)
            if not text:
                continue

            texts.append(text)
            labels.append(topics[0])

    return texts, labels

def main(data_dir: str = ".", out_path: str = "datasets/Reuters.mat", top_k: int = 10,
         word_dim: int = 256, char_dim: int = 256, max_features: int = 20000):

    texts, topics = load_reuters(data_dir, single_label_only=True)
    print(f"Loaded docs (single-topic only): {len(texts)}")

    # choose top-K topics
    freq = Counter(topics)
    top = [t for t, _ in freq.most_common(top_k)]
    keep = set(top)

    filt_texts = []
    filt_topics = []
    for t, lab in zip(texts, topics):
        if lab in keep:
            filt_texts.append(t)
            filt_topics.append(lab)

    topic_to_id = {t: i for i, t in enumerate(top)}
    y = np.array([topic_to_id[t] for t in filt_topics], dtype=np.int32).reshape(-1, 1)
    print(f"After filtering to top-{top_k} topics: {len(filt_texts)} samples, K={len(top)}")

    # View 1: word TF-IDF -> SVD
    word_vec = TfidfVectorizer(
      stop_words="english",
      lowercase=True,
      max_df=0.9,
      min_df=2,
      max_features=50000,
    )   
    Xw = word_vec.fit_transform(filt_texts)
    svd_w = TruncatedSVD(n_components=min(word_dim, Xw.shape[1]-1), random_state=0)
    Xw = svd_w.fit_transform(Xw).astype(np.float32)
    Xw = StandardScaler().fit_transform(Xw).astype(np.float32)

    # View 2: char TF-IDF -> SVD
    char_vec = TfidfVectorizer(
      analyzer="char_wb",
      ngram_range=(3, 5),
      lowercase=True,
      max_df=0.95,
      min_df=2,
      max_features=80000,
    )
    Xc = char_vec.fit_transform(filt_texts)
    svd_c = TruncatedSVD(n_components=min(char_dim, Xc.shape[1]-1), random_state=0)
    Xc = svd_c.fit_transform(Xc).astype(np.float32)
    Xc = StandardScaler().fit_transform(Xc).astype(np.float32)

    X_cell = np.empty((1, 2), dtype=object)
    X_cell[0, 0] = Xw
    X_cell[0, 1] = Xc

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    topic_names_arr = np.array(top, dtype=object).reshape(-1, 1)
    savemat(out_path, {"X": X_cell, "Y": y, "topic_names": topic_names_arr})
    print(f"Saved to {out_path}")
    print("Top topics:")
    for t in top:
        print(f"  {topic_to_id[t]}: {t} ({freq[t]})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out", default="datasets/Reuters.mat")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--word_dim", type=int, default=256)
    ap.add_argument("--char_dim", type=int, default=256)
    ap.add_argument("--max_features", type=int, default=20000)
    args = ap.parse_args()

    main(args.data_dir, args.out, args.top_k, args.word_dim, args.char_dim, args.max_features)

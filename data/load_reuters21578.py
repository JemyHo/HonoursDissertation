from pathlib import Path
from collections import Counter
import numpy as np

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _iter_reuters_docs(reuters_dir: str, splits=("TRAIN", "TEST")):
    """
    Iterate over Reuters-21578 SGML documents.
    Yields: (text, topics, split_tag)
    """
    reuters_dir = Path(reuters_dir)
    sgm_files = sorted(reuters_dir.glob("reut2-*.sgm"))
    if not sgm_files:
        raise FileNotFoundError(f"No reut2-*.sgm found in: {reuters_dir}")

    for fp in sgm_files:
        soup = BeautifulSoup(fp.read_text(errors="ignore"), "html.parser")
        for r in soup.find_all("reuters"):
            split_tag = (r.get("lewissplit") or "").upper()
            if split_tag and split_tag not in splits:
                continue

            # Topics (can be multi-label)
            topics_node = r.find("topics")
            topics = []
            if topics_node:
                topics = [d.get_text(strip=True) for d in topics_node.find_all("d")]

            title = r.find("title")
            body = r.find("body")
            text = " ".join([
                title.get_text(" ", strip=True) if title else "",
                body.get_text(" ", strip=True) if body else ""
            ]).strip()

            if text:
                yield text, topics, split_tag


def load_reuters21578_kmeans(
    reuters_dir="Dataset/Reuters",
    splits=("TRAIN",),          # use TRAIN only by default (smaller + standard-ish)
    top_k_topics=10,            # cluster into top-K topics
    svd_dim=300,                # make features dense for your pipeline
    max_features_word=50000,
    max_features_char=80000,
    random_state=0,
):
    """
    Build 2-view feature matrices from Reuters text and return:
      views: [X_word, X_char]  (dense float32 matrices)
      y_true: integer labels for evaluation
      topic_names: list of chosen topic strings (id -> name)
    Notes:
      - Reuters is multi-label; for ACC/NMI/ARI we keep SINGLE-TOPIC docs only.
      - Clustering is unsupervised: y_true is used only for scoring.
    """
    texts = []
    single_topics = []

    # 1) Parse SGMs and keep docs with exactly one topic label
    for text, topics, _split in _iter_reuters_docs(reuters_dir, splits=splits):
        if len(topics) == 1:
            texts.append(text)
            single_topics.append(topics[0])

    if not texts:
        raise ValueError("No single-topic documents found. Check your files/splits.")

    # 2) Pick top-K most frequent topics (reduces imbalance + defines K)
    freq = Counter(single_topics)
    topic_names = [t for t, _ in freq.most_common(top_k_topics)]

    # 3) Filter to those topics and map to 0..K-1
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    keep_idx = [i for i, t in enumerate(single_topics) if t in topic_to_id]

    texts = [texts[i] for i in keep_idx]
    y_true = np.array([topic_to_id[single_topics[i]] for i in keep_idx], dtype=np.int64)

    if len(texts) < top_k_topics * 20:
        # Not fatal, but a warning sign (too few docs per cluster)
        print(f"[warn] Only {len(texts)} docs for top-{top_k_topics} topics.")

    # 4) View A: word TF-IDF -> SVD -> dense
    word_vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.9,
        min_df=2,
        max_features=max_features_word,
    )
    Xw_sparse = word_vec.fit_transform(texts)
    Xw = TruncatedSVD(n_components=svd_dim, random_state=random_state).fit_transform(Xw_sparse).astype(np.float32)

    # 5) View B: char TF-IDF -> SVD -> dense (different signal vs word tokens)
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=max_features_char,
    )
    Xc_sparse = char_vec.fit_transform(texts)
    Xc = TruncatedSVD(n_components=svd_dim, random_state=random_state).fit_transform(Xc_sparse).astype(np.float32)

    views = [Xw, Xc]
    return views, y_true, topic_names

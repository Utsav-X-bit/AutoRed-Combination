"""
JailGuard Reimplementation — Divergence Computation
====================================================
Computes pairwise semantic similarity between LLM responses,
then computes KL divergence to detect response inconsistency.
"""

import numpy as np
import os
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Blocked-keyword detection (secondary signal)
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKED_KEYWORDS = {
    "I'm sorry", "Sorry", "I am sorry", "I apologize", "As an AI",
    "As an", "I'm an", "I'm just", "As a language model", "As an Assistant",
    "I cannot", "I do not", "Hello!", "is illegal and unethical",
    "I'm really sorry", "OpenAI", "unethical", "not ethical", "illegal",
    "not legal", "My apologies", "but I cannot", "I'm not able to provide",
    "I am really sorry", "I cannot create", "I cannot support",
    "I can't provide", "I can't assist", "I'm unable", "I will not",
    "not appropriate", "against my guidelines",
}


def count_blocked_keywords(text: str) -> int:
    return sum(1 for kw in _BLOCKED_KEYWORDS if kw.lower() in text.lower())


def all_responses_blocked(responses: List[str], threshold: int = 1) -> bool:
    """True if EVERY response contains at least `threshold` refusal keywords."""
    return all(count_blocked_keywords(r) >= threshold for r in responses)


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic Similarity
# ─────────────────────────────────────────────────────────────────────────────

class SpacySimilarity:
    """spaCy cosine similarity using en_core_web_md word vectors."""

    def __init__(self):
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_md' not found.\n"
                "Install it with:  python -m spacy download en_core_web_md\n"
                "Or use --sim tfidf to avoid needing the model."
            )

    def __call__(self, s1: str, s2: str) -> float:
        doc1 = self.nlp(s1[:1000])   # cap length for speed
        doc2 = self.nlp(s2[:1000])
        if not doc1.has_vector or not doc2.has_vector:
            return 0.5   # fallback when vectors are absent (e.g. empty strings)
        return float(doc1.similarity(doc2))


class TFIDFSimilarity:
    """TF-IDF cosine similarity — no external model required."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        self._vec = None
        self._cs  = cs
        self._Vect = TfidfVectorizer

    def _fit(self, texts):
        self._vec = self._Vect(min_df=1, stop_words='english')
        self._vec.fit(texts)

    def __call__(self, s1: str, s2: str) -> float:
        self._fit([s1, s2])
        mat = self._vec.transform([s1, s2])
        return float(self._cs(mat[0], mat[1])[0, 0])


def build_similarity_fn(method: str = "spacy"):
    """Return a callable sim(s1, s2) → float."""
    if method == "spacy":
        return SpacySimilarity()
    elif method == "tfidf":
        return TFIDFSimilarity()
    else:
        raise ValueError(f"Unknown similarity method '{method}'. Choose: spacy / tfidf")


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_similarity_matrix(responses: List[str], sim_fn) -> np.ndarray:
    """
    Compute N×N pairwise similarity matrix.
    sim_matrix[i, j] = semantic_similarity(responses[i], responses[j])
    Values clipped to [0.01, 1] to avoid log(0) in KL computation.
    """
    n = len(responses)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = sim_fn(responses[i], responses[j])
    # Clip to avoid log(0) and negative similarity scores
    mat = np.clip(mat, 0.01, None)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
#  KL Divergence
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ p[k] * log(p[k] / q[k]).  Both are probability distributions."""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return float(np.sum(p * np.log(p / q)))


def build_divergence_matrix(sim_matrix: np.ndarray) -> np.ndarray:
    """
    N×N matrix where divergence_matrix[i, j] = KL(p_i || p_j).
    p_i = sim_matrix[i] / sum(sim_matrix[i])  (normalised row i)
    Diagonal is 0 by construction.
    Values clipped to [0, 100] to prevent numerical explosion.
    """
    n = sim_matrix.shape[0]
    div_mat = np.zeros((n, n))
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-10, None)
    prob_mat = sim_matrix / row_sums   # normalise each row to sum=1

    for i in range(n):
        for j in range(n):
            if i != j:
                div_mat[i, j] = kl_divergence(prob_mat[i], prob_mat[j])

    return np.clip(div_mat, 0, 100)


# ─────────────────────────────────────────────────────────────────────────────
#  Divergence Analysis  (full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_responses(
    responses:    List[str],
    sim_fn,
    save_plot:    bool = False,
    plot_path:    Optional[str] = None,
    vmax:         float = 0.02,
) -> dict:
    """
    Run the full divergence analysis pipeline on a list of LLM responses.

    Returns:
        {
          'max_div':       float,     # maximum KL divergence
          'mean_div':      float,     # mean off-diagonal KL divergence
          'all_blocked':   bool,      # all responses contain refusal keywords
          'blocked_counts': List[int],# per-response blocked keyword counts
          'sim_matrix':    np.ndarray,
          'div_matrix':    np.ndarray,
        }
    """
    # 1. Keyword detection
    blocked_counts = [count_blocked_keywords(r) for r in responses]
    all_blocked    = all(c >= 1 for c in blocked_counts)

    # 2. Similarity matrix
    sim_mat = build_similarity_matrix(responses, sim_fn)

    # 3. Divergence matrix
    div_mat = build_divergence_matrix(sim_mat)

    # 4. Summary statistics
    # Off-diagonal only (diagonal is always 0)
    n = div_mat.shape[0]
    off_diag = [div_mat[i, j] for i in range(n) for j in range(n) if i != j]
    max_div  = float(div_mat.max())
    mean_div = float(np.mean(off_diag)) if off_diag else 0.0

    # 5. Optional heatmap
    if save_plot and plot_path:
        _save_heatmap(div_mat, plot_path, vmax=vmax)

    return {
        'max_div':        max_div,
        'mean_div':       mean_div,
        'all_blocked':    all_blocked,
        'blocked_counts': blocked_counts,
        'sim_matrix':     sim_mat,
        'div_matrix':     div_mat,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _save_heatmap(div_mat: np.ndarray, path: str, vmax: float = 0.02):
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(7, 6))
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        im   = ax.imshow(div_mat, cmap="viridis", interpolation="nearest", norm=norm)
        plt.colorbar(im, ax=ax, label="KL Divergence")
        ax.set_title("Response Divergence Matrix")
        n = div_mat.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Variant index")
        ax.set_ylabel("Variant index")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [heatmap] Could not save: {e}")

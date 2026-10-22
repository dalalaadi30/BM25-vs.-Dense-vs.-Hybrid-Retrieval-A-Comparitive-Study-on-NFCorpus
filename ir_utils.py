"""Small, dependency-light helpers shared by the notebook and the unit tests.

Everything here is pure Python / NumPy (+ SciPy for Welch's t-test) so it can be
tested without downloading the dataset or the encoder.
"""
import math
import re
from collections import defaultdict

import numpy as np


def simple_tokenize(text):
    """Lower-case alphanumeric tokenizer (no stemming, no stop-word removal)."""
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(rank_lists, c=60, k=10):
    """Fuse ranked doc-id lists with RRF: score(d) = sum_s 1 / (c + rank_s(d)).

    Ranks are 1-based. Ties in fused score are broken by first appearance
    (stable sort), so the output is deterministic.
    """
    scores = defaultdict(float)
    for ranked_list in rank_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (c + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ranked[:k]]


# --------------------------------------------------------------------------
# Metrics (all take a ranked list of doc ids)
# --------------------------------------------------------------------------
def recall_at_k(retrieved, relevant_set, k):
    if not relevant_set:
        return None
    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved, relevant_set, k=None):
    """1 / rank of the first relevant doc; 0 if none in the (optionally cut) list."""
    for rank, doc_id in enumerate(retrieved[:k] if k else retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, graded_rels, k=10):
    """nDCG@k with linear gain (as in BEIR / pytrec_eval). graded_rels: {doc: rel}."""
    gains = [max(graded_rels.get(d, 0), 0) for d in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((r for r in graded_rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def lexical_overlap(query_text, doc_text):
    """Fraction of the query's unique tokens that also occur in the document."""
    q = set(simple_tokenize(query_text))
    if not q:
        return 0.0
    return len(q & set(simple_tokenize(doc_text))) / len(q)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def permutation_test(a, b, n_perm=10000, seed=42):
    """Two-sided permutation test on the difference of means. Returns (diff, p).

    p uses the +1 correction (never reports exactly 0).
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = a.mean() - b.mean()
    pooled, n1 = np.concatenate([a, b]), len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:n1].mean() - pooled[n1:].mean()) >= abs(observed):
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def bootstrap_diff_ci(a, b, n_boot=10000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for mean(a) - mean(b) (independent groups)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
                      for _ in range(n_boot)])
    return tuple(np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def paired_bootstrap(x, y, n_boot=10000, seed=42, alpha=0.05):
    """Paired bootstrap over queries for mean(x) - mean(y).

    Returns (mean_diff, (ci_low, ci_high), two_sided_p) where p is the share of
    resampled differences on the opposite side of zero from the observed one,
    doubled (capped at 1) and with a +1 correction.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    obs = d.mean()
    opp = (boots <= 0).sum() if obs > 0 else (boots >= 0).sum()
    p = min(1.0, 2 * (opp + 1) / (n_boot + 1))
    return obs, (lo, hi), p

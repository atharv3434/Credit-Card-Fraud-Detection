"""
Business-cost evaluation (numpy/pandas, unit-tested).

Economics: flagging a transaction costs `review_cost` (an analyst looks at it). If it was fraud we
prevent the loss (its amount); if not we just wasted the review cost. Doing nothing saves $0.
Simplification: each transaction is judged independently (blocking one in a burst would also
prevent the later ones, which would make real-world savings higher).

"""
import numpy as np


def savings(amount, label, flag, review_cost: float = 5.0) -> float:
    amount, label, flag = (np.asarray(a) for a in (amount, label, flag))
    flag = flag.astype(bool)
    return float(amount[flag & (label == 1)].sum() - review_cost * flag.sum())


def best_threshold(amount, label, score, review_cost: float = 5.0):
    """Score cut-off maximising net savings. Returns (threshold, savings). threshold=inf => flag nothing."""
    amount, label, score = (np.asarray(a, float) for a in (amount, label, score))
    order = np.argsort(-score, kind="stable")
    gain = np.cumsum(amount[order] * label[order] - review_cost)
    # only cut between distinct scores so ties are never split
    valid = np.r_[score[order][1:] != score[order][:-1], True]
    gain = np.where(valid, gain, -np.inf)
    i = int(np.argmax(gain))
    if gain[i] <= 0:
        return float("inf"), 0.0
    return float(score[order][i]), float(gain[i])


def recall_at_budget(label, score, frac: float) -> float:
    """Share of all frauds found when analysts review only the top `frac` of transactions by score."""
    label, score = np.asarray(label), np.asarray(score, float)
    k = max(1, int(round(frac * len(label))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(label[top].sum() / max(label.sum(), 1))


def pr_curve(label, score):
    label, score = np.asarray(label), np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(label[order]); n = np.arange(1, len(label) + 1)
    return tp / n, tp / max(label.sum(), 1)            # precision, recall


def average_precision(label, score) -> float:
    p, r = pr_curve(label, score)
    r_prev = np.r_[0.0, r[:-1]]
    return float(np.sum((r - r_prev) * p))

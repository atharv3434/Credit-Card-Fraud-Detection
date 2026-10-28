import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src import costs  # noqa: E402

vp, tp, imp = pd.read_csv(C.VAL_PREDS), pd.read_csv(C.TEST_PREDS), pd.read_csv(C.IMPORTANCE)
BEST = "gbt_sqrt_weight"
NAMES = {"logreg_balanced": "Logistic regression", "gbt_unweighted": "GBT (unweighted)", BEST: "GBT (weighted)"}
thr, _ = costs.best_threshold(vp.amount, vp.label, vp[f"p_{BEST}"], C.REVIEW_COST)     # chosen on validation only
rule = ((tp.foreign == 1) & (tp.amount > 300)) | (tp.txns_10m >= 5)
flag = tp[f"p_{BEST}"] >= thr

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# 1. precision-recall
for n, c in zip(NAMES, ["#8895a7", "#4c8bf5", "#1a56c9"]):
    p, r = costs.pr_curve(tp.label, tp[f"p_{n}"])
    ax[0, 0].plot(r, p, color=c, label=f"{NAMES[n]} (AP={costs.average_precision(tp.label, tp[f'p_{n}']):.3f})")
rp = tp[rule & (tp.label == 1)].shape[0]
ax[0, 0].scatter(rp / tp.label.sum(), rp / rule.sum(), color="#d9534f", s=70, zorder=5, label="Hand-written rule")
fr = tp[flag & (tp.label == 1)].shape[0]
ax[0, 0].scatter(fr / tp.label.sum(), fr / flag.sum(), color="black", marker="*", s=180, zorder=5, label="Chosen operating point")
ax[0, 0].axhline(tp.label.mean(), color="grey", ls=":", label=f"Random ({tp.label.mean():.1%})")
ax[0, 0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test set)"); ax[0, 0].legend(fontsize=8, loc="lower left")

# 2. net savings vs transactions reviewed
order = np.argsort(-tp[f"p_{BEST}"].values)
gain = np.cumsum(tp.amount.values[order] * tp.label.values[order] - C.REVIEW_COST)
share = np.arange(1, len(tp) + 1) / len(tp) * 100
ax[0, 1].plot(share, gain, color="#1a56c9", label="GBT: review in score order")
ax[0, 1].scatter(flag.mean() * 100, costs.savings(tp.amount, tp.label, flag, C.REVIEW_COST), color="black", marker="*", s=180, zorder=5, label="Chosen threshold (from validation)")
ax[0, 1].scatter(rule.mean() * 100, costs.savings(tp.amount, tp.label, rule, C.REVIEW_COST), color="#d9534f", s=70, zorder=5, label="Hand-written rule")
ax[0, 1].set_xlim(0, 8); ax[0, 1].set_ylim(-2000, gain[share <= 8].max() * 1.12); ax[0, 1].axhline(0, color="grey", lw=0.8)
ax[0, 1].set(xlabel="% of transactions sent to review", ylabel="Net savings ($)", title="Net savings vs review volume (test set)"); ax[0, 1].legend(fontsize=8)

# 3. recall by fraud type
fr_df = tp[tp.label == 1].assign(caught=flag[tp.label == 1])
g = fr_df.groupby("fraud_type").caught.agg(["mean", "size"])
bars = ax[1, 0].bar(g.index, g["mean"], color=["#4c8bf5", "#1a56c9", "#8895a7"])
for b, (m, n) in zip(bars, zip(g["mean"], g["size"])):
    ax[1, 0].text(b.get_x() + b.get_width() / 2, m, f"{m:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
ax[1, 0].set(ylim=(0, 1.2), ylabel="Recall at chosen threshold", title="Recall by fraud pattern (error analysis)")

# 4. feature importance
top = imp.head(10).iloc[::-1]
ax[1, 1].barh(top.feature, top.importance, color="#1a56c9"); ax[1, 1].set_title("GBT feature importance")
plt.tight_layout(); plt.savefig(C.ROOT / "reports" / "fraud_results.png", dpi=125)
print("saved; chosen threshold", round(thr, 3))

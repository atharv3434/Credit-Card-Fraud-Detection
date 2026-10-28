"""Train -> select on validation -> evaluate ONCE on test, in precision/recall terms AND dollars."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src import config as C  # noqa: E402
from src import costs  # noqa: E402
from src.features import FEATURES, build_features  # noqa: E402
from src.spark_session import get_spark  # noqa: E402
from src.splits import split  # noqa: E402

pd.set_option("display.width", 170)
spark = get_spark()
tx = (spark.read.csv(C.TX_CSV, header=True)
      .select("txn_id", F.col("user_id").cast("int"), F.to_timestamp("ts").alias("ts"), F.col("amount").cast("double"),
              F.col("merchant_id").cast("int"), "merchant_category", "channel", "country",
              F.coalesce("device_id", F.lit("")).alias("device_id"), F.col("is_fraud").cast("int"), "fraud_type"))
users = spark.read.csv(C.USERS_CSV, header=True).select(F.col("user_id").cast("int"), "home_country", F.col("account_age_days").cast("int"))
from pyspark import StorageLevel  # noqa: E402
# one cached copy only (spills to disk); the splits are cheap filters on top of it. Keeps memory low on small machines.
feats = build_features(tx, users).persist(StorageLevel.MEMORY_AND_DISK)
train, val, test = split(feats)
for name, d in [("train", train), ("val", val), ("test", test)]:
    n, f = d.count(), d.agg(F.sum("is_fraud")).first()[0]
    print(f"{name:<5} rows={n:>8,}  fraud={f:>5,} ({f / n:.2%})", flush=True)

pos = train.agg(F.sum("is_fraud")).first()[0]; neg = train.count() - pos
W_BAL, W_SQRT = neg / pos, float(np.sqrt(neg / pos))
print(f"class ratio neg:pos = {neg / pos:.0f}:1  -> balanced weight {W_BAL:.0f}, sqrt weight {W_SQRT:.1f}", flush=True)
asm = lambda: VectorAssembler(inputCols=FEATURES, outputCol="features")
gbt = lambda w=None: GBTClassifier(featuresCol="features", labelCol="label", maxIter=40, maxDepth=5, stepSize=0.15,
                                   subsamplingRate=0.8, seed=C.SEED, **({"weightCol": w} if w else {}))
MODELS = {
    "logreg_balanced": (Pipeline(stages=[asm(), LogisticRegression(labelCol="label", weightCol="w_bal", regParam=1e-3, maxIter=100)]), "w_bal"),
    "gbt_unweighted": (Pipeline(stages=[asm(), gbt()]), None),
    "gbt_sqrt_weight": (Pipeline(stages=[asm(), gbt("w_sqrt")]), "w_sqrt"),
}
add_w = lambda d: d.withColumn("w_bal", F.when(F.col("label") == 1, W_BAL).otherwise(1.0)).withColumn("w_sqrt", F.when(F.col("label") == 1, W_SQRT).otherwise(1.0))
train_w = add_w(train)

pr, roc = BinaryClassificationEvaluator(metricName="areaUnderPR"), BinaryClassificationEvaluator(metricName="areaUnderROC")
fitted, base = {}, {"val": val.select("txn_id", "user_id", "ts", "amount", "label", "fraud_type", "channel", "foreign", "txns_10m"),
                    "test": test.select("txn_id", "user_id", "ts", "amount", "label", "fraud_type", "channel", "foreign", "txns_10m")}
metrics = []
for name, (pipe, _) in MODELS.items():
    print(f"fitting {name} ...", flush=True)
    m = pipe.fit(train_w); fitted[name] = m
    for split_name, d in [("val", val), ("test", test)]:
        out = m.transform(d)
        metrics.append({"model": name, "split": split_name, "PR_AUC": pr.evaluate(out), "ROC_AUC": roc.evaluate(out)})
        scores = out.select("txn_id", vector_to_array("probability")[1].alias(f"p_{name}"))
        base[split_name] = base[split_name].join(scores, "txn_id")
print("\n=== SPARK METRICS (PR-AUC is the honest one under imbalance; a coin flip scores ~0.007) ===")
print(pd.DataFrame(metrics).pivot(index="model", columns="split", values=["PR_AUC", "ROC_AUC"]).round(4), flush=True)

C.OUT.mkdir(parents=True, exist_ok=True)
cols = lambda d: pd.DataFrame([r.asDict() for r in d.collect()])
vp, tp = cols(base["val"]), cols(base["test"])
vp.to_csv(C.VAL_PREDS, index=False); tp.to_csv(C.TEST_PREDS, index=False)
imp = pd.DataFrame({"feature": FEATURES, "importance": fitted["gbt_sqrt_weight"].stages[-1].featureImportances.toArray()}).sort_values("importance", ascending=False)
imp.to_csv(C.IMPORTANCE, index=False)
spark.stop()

# ---------------- model + threshold selection on VALIDATION only ----------------
names = list(MODELS)
val_ap = {n: costs.average_precision(vp.label, vp[f"p_{n}"]) for n in names}
best = max(val_ap, key=val_ap.get)
thr, val_gain = costs.best_threshold(vp.amount, vp.label, vp[f"p_{best}"], C.REVIEW_COST)
print(f"\nselected on validation: model={best} (PR-AUC {val_ap[best]:.3f}), threshold={thr:.3f}, val savings=${val_gain:,.0f}")

rule = lambda d: ((d.foreign == 1) & (d.amount > 300)) | (d.txns_10m >= 5)
total_fraud = tp[tp.label == 1].amount.sum()
print(f"\n=== TEST SET: {len(tp):,} transactions, {int(tp.label.sum())} frauds worth ${total_fraud:,.0f} ===")
rows = [("do nothing", np.zeros(len(tp), bool)), ("hand-written rule", rule(tp).values),
        (f"{best} @ val threshold", (tp[f"p_{best}"] >= thr).values)]
res = []
for label, flag in rows:
    caught = tp[flag & (tp.label == 1)]
    res.append({"strategy": label, "flagged": int(flag.sum()), "frauds_caught": len(caught),
                "recall": len(caught) / tp.label.sum(), "precision": len(caught) / max(flag.sum(), 1),
                "fraud_$_prevented": caught.amount.sum(), "review_cost": C.REVIEW_COST * flag.sum(),
                "net_savings": costs.savings(tp.amount, tp.label, flag, C.REVIEW_COST)})
print(pd.DataFrame(res).round(3).to_string(index=False))
_, oracle = costs.best_threshold(tp.amount, tp.label, tp[f"p_{best}"], C.REVIEW_COST)
print(f"(reference only: best possible threshold chosen with test hindsight would save ${oracle:,.0f})")

print("\n=== recall when analysts can review only the top X% of transactions ===")
print(pd.DataFrame({"review budget": ["0.25%", "0.5%", "1%", "2%"],
                    **{n: [costs.recall_at_budget(tp.label, tp[f"p_{n}"], f) for f in (.0025, .005, .01, .02)] for n in names}}).round(3).to_string(index=False))

flag = (tp[f"p_{best}"] >= thr)
fr = tp[tp.label == 1]
print("\n=== recall by fraud type at the chosen threshold ===")
print(fr.assign(caught=flag[fr.index]).groupby("fraud_type").agg(frauds=("caught", "size"), recall=("caught", "mean"), avg_amount=("amount", "mean")).round(3))
print("\n=== top features ===")
print(imp.head(8).round(3).to_string(index=False))

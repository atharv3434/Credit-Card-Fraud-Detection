# Credit-Card Fraud Detection (PySpark) - evaluated in dollars, not just AUC

Detects fraudulent card transactions (0.7% of traffic) using behavioural features built with Spark
window functions, gradient-boosted trees, a **time-based split with a label-delay gap**, and a
business-cost evaluation that chooses the alert threshold from review cost vs. fraud loss.

## Quick start
```bash
pip install -r requirements.txt     # needs Java 17+, ~4 GB RAM
python scripts/generate_data.py     # 288K transactions, 4,000 users, 90 days, 3 fraud patterns
python scripts/train.py             # ~8 min locally: features, 3 models, selection, test evaluation
python scripts/report.py            # reports/fraud_results.png
pytest -q                           # 7 tests incl. no-future-leakage proof
```

## Results (test set: 25,481 transactions, 197 frauds worth $40,129; review cost $5/alert)
| Strategy | Flagged | Frauds caught | Recall | Precision | **Net savings** |
|---|---|---|---|---|---|
| Do nothing | 0 | 0 | 0% | - | $0 |
| Hand-written rule | 99 | 75 | 38% | 76% | $18,952 |
| **GBT @ threshold chosen on validation** | 373 | 189 | **96%** | 51% | **$35,055** |

- The validation-chosen threshold got **99.6% of the best achievable** savings ($35,180, found with
  test hindsight), so it generalised; the test set made no choices.
- **PR-AUC 0.925** (GBT, weighted) vs 0.815 (logistic regression). A random scorer gets ~0.008.
  ROC-AUC is 0.998 for the GBT, which looks great but says little at 0.7% prevalence; PR-AUC and
  dollar savings are the honest measures.
- Review-budget view: reviewing just the top **1%** of transactions finds 91% of frauds.
- Recall by pattern at the chosen threshold: card testing 100%, account takeover 95%, stolen card 87%.

## Design decisions worth knowing
- **Temporal split with a 7-day gap** (train days 0-56 | val 63-75 | test 82-90). Fraud labels
  (chargebacks) arrive about a week late, so a model deployed on day N never sees labels from
  days N-7..N. A random split would be wildly optimistic.
- **Leak-free features**: velocity (txns in 10 min / 1 h / 24 h), amount vs. the user's own history,
  new device / new country, time since previous transaction. Every window ends strictly before the
  current row. Tests prove that rewriting *later* transactions leaves a row's features unchanged.
  No target encoding or merchant fraud rates (they would leak labels).
- **Class imbalance (153:1)**: compared logistic regression with full balancing, unweighted GBT, and
  GBT with square-root class weights; selected by validation PR-AUC.
- **Cost-based thresholding**: flag a transaction if the expected gain beats the $5 review cost; the
  threshold is picked on validation only, then applied once to test.

## Limitations (please read)
- **Synthetic data with clean, separable patterns.** Card-testing bursts are trivially visible in
  `txns_1h`, the top feature, so near-perfect recall there is a property of the simulator. Real fraud
  adapts to defences; expect lower numbers. I added "hard" fraud (home country, known device) and
  legitimate noise (travel, big purchases), but real data is messier.
- **Small test set: 197 frauds.** A few transactions move recall by ~0.5 pt. The weighted vs. unweighted
  GBT gap (PR-AUC 0.925 vs 0.915) is within noise; one seed, no confidence intervals.
- **Simplified economics**: each transaction is judged independently, so blocking one burst
  transaction would also prevent later ones (savings are understated). Real reviews also add customer
  friction costs from false positives (51% precision at this threshold), which are not modelled.
- **No drift**: val and test come from the same generator; real fraud rings change tactics, so
  production needs monitoring and retraining.
- **Probabilities are not calibrated** (class weighting distorts them); only the ranking and the
  validation-chosen threshold are meaningful.
- Spark ran in local mode; the same code scales out but this was not benchmarked on a cluster.

## Ideas to extend
Graph features (shared devices / merchants across users), streaming scoring with the IoT project's
pattern, SHAP-style reason codes for analysts, drift monitoring (PSI), cost-sensitive learning,
and a champion/challenger evaluation.

## Layout
```
scripts/generate_data.py  simulator + fraud patterns    src/features.py  window-function features
scripts/train.py          train, select, evaluate       src/splits.py    temporal split with gap
scripts/report.py         charts                        src/costs.py     savings, thresholds, PR, recall@budget
tests/test_fraud.py                                      reports/         fraud_results.png
```

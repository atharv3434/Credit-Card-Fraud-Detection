from datetime import datetime, timedelta

import numpy as np
import pytest
from pyspark.sql import SparkSession

from src import config as C
from src import costs
from src.features import build_features


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()


T0 = datetime(2025, 1, 10, 12, 0, 0)
COLS = ["txn_id", "user_id", "ts", "amount", "merchant_id", "merchant_category", "channel", "country",
        "device_id", "is_fraud", "fraud_type"]


def _tx(spark, spec):
    """spec rows: (seconds_offset, amount, channel, country, device)"""
    rows = [(f"T{i:03d}", 1, T0 + timedelta(seconds=s), float(a), 1, "grocery", ch, ct, d, 0, "")
            for i, (s, a, ch, ct, d) in enumerate(spec)]
    tx = spark.createDataFrame(rows, COLS)
    users = spark.createDataFrame([(1, "US", 300)], ["user_id", "home_country", "account_age_days"])
    return build_features(tx, users)


def _by_id(df):
    return {r.txn_id: r for r in df.collect()}


def test_velocity_counts_only_earlier_rows_in_window(spark):
    spec = [(0, 10, "pos", "US", ""), (100, 10, "pos", "US", ""), (200, 10, "pos", "US", ""), (5000, 10, "pos", "US", "")]
    r = _by_id(_tx(spark, spec))
    assert r["T000"].txns_1h == 0 and r["T000"].n_prior == 0
    assert r["T002"].txns_1h == 2 and r["T002"].txns_10m == 2          # counts T000, T001, not itself
    assert r["T003"].txns_1h == 0 and r["T003"].txns_24h == 3          # >1h ago but within 24h


def test_features_ignore_the_future(spark):
    base = [(0, 10, "pos", "US", ""), (100, 20, "online", "US", "D1"), (200, 30, "pos", "US", "")]
    tampered = base[:2] + [(200, 99999, "online", "FR", "D-NEW")] + [(300, 5000, "online", "FR", "D-NEW")]
    a, b = _by_id(_tx(spark, base)), _by_id(_tx(spark, tampered))
    for f in ["amount_ratio", "amount_z", "n_prior", "txns_10m", "txns_1h", "txns_24h", "log_amt_sum_24h",
              "countries_24h", "log_secs_since_prev", "new_device", "new_country"]:
        assert a["T001"][f] == pytest.approx(b["T001"][f]), f"{f} at T001 depends on later rows"


def test_new_device_and_country_flags(spark):
    spec = [(0, 10, "online", "US", "A"), (100, 10, "online", "US", "A"), (200, 10, "online", "US", "B"),
            (300, 10, "online", "FR", "B"), (400, 10, "pos", "FR", "")]
    r = _by_id(_tx(spark, spec))
    assert [r[f"T00{i}"].new_device for i in range(5)] == [0, 0, 1, 0, 0]    # first-ever device isn't 'new'; POS has none
    assert [r[f"T00{i}"].new_country for i in range(5)] == [0, 0, 0, 1, 0]
    assert r["T003"].foreign == 1 and r["T000"].foreign == 0


def test_amount_ratio_uses_prior_mean(spark):
    r = _by_id(_tx(spark, [(0, 10, "pos", "US", ""), (100, 30, "pos", "US", ""), (200, 100, "pos", "US", "")]))
    assert r["T002"].amount_ratio == pytest.approx(100 / (20 + 1))


def test_temporal_splits_leave_a_label_maturity_gap_and_do_not_overlap():
    from src.splits import bounds
    tr, va, te = bounds(C.TRAIN_DAYS), bounds(C.VAL_DAYS), bounds(C.TEST_DAYS)
    assert tr[1] + timedelta(days=7) <= va[0] and va[1] + timedelta(days=7) <= te[0]


def test_savings_and_best_threshold():
    amount, label = [500, 5, 100, 50], [1, 0, 1, 0]
    assert costs.savings(amount, label, [1, 1, 0, 0], review_cost=5) == pytest.approx(500 - 10)
    thr, gain = costs.best_threshold(amount, label, [0.9, 0.8, 0.7, 0.1], review_cost=5)
    assert thr == pytest.approx(0.7) and gain == pytest.approx(500 + 100 - 15)   # flag 3, skip the last
    assert costs.best_threshold([1, 1], [0, 0], [0.9, 0.1], 5) == (float("inf"), 0.0)   # never worth flagging


def test_pr_and_budget_metrics():
    y, s = [1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.2, 0.1]
    assert costs.average_precision(y, s) == pytest.approx(1 * 0.5 + (2 / 3) * 0.5)
    assert costs.recall_at_budget(y, s, 0.2) == pytest.approx(0.5)               # top 1 of 5 -> found 1 of 2 frauds
    assert costs.average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)

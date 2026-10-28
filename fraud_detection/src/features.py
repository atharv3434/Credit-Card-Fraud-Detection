"""Behavioural features with Spark window functions.

LEAKAGE RULES
  * Every history-based feature for a transaction uses only that user's EARLIER transactions
    (windows end at -1 / exclude the current row). Nothing from the future, no labels.
  * No target encoding, no fraud-rate-by-merchant: those would leak labels.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

CATS = ["grocery", "restaurant", "fuel", "electronics", "travel", "digital_goods", "gift_cards", "jewelry", "clothing", "utilities"]

FEATURES = (
    ["log_amount", "amount_ratio", "amount_z", "n_prior", "txns_10m", "txns_1h", "txns_24h", "log_amt_sum_24h",
     "countries_24h", "log_secs_since_prev", "is_online", "foreign", "new_device", "new_country",
     "hour", "is_night", "dow", "account_age_days"]
    + [f"cat_{c}" for c in CATS]
)


def build_features(tx: DataFrame, users: DataFrame) -> DataFrame:
    df = (tx.join(F.broadcast(users), "user_id", "left")
          .withColumn("ts_unix", F.col("ts").cast("long")))

    by_order = Window.partitionBy("user_id").orderBy("ts_unix", "txn_id")
    hist = by_order.rowsBetween(Window.unboundedPreceding, -1)                    # strictly earlier rows

    def last(seconds: int):                                                       # earlier rows within N seconds
        return Window.partitionBy("user_id").orderBy("ts_unix").rangeBetween(-seconds, -1)

    df = (df.withColumn("n_prior", F.count(F.lit(1)).over(hist))
          .withColumn("mean_prior", F.avg("amount").over(hist))
          .withColumn("std_prior", F.stddev("amount").over(hist))
          .withColumn("txns_10m", F.count(F.lit(1)).over(last(600)))
          .withColumn("txns_1h", F.count(F.lit(1)).over(last(3600)))
          .withColumn("txns_24h", F.count(F.lit(1)).over(last(86400)))
          .withColumn("amt_sum_24h", F.coalesce(F.sum("amount").over(last(86400)), F.lit(0.0)))
          .withColumn("countries_24h", F.size(F.collect_set("country").over(last(86400))))
          .withColumn("secs_since_prev", F.col("ts_unix") - F.lag("ts_unix").over(by_order)))

    first_dev = F.row_number().over(Window.partitionBy("user_id", "device_id").orderBy("ts_unix", "txn_id")) == 1
    first_ctry = F.row_number().over(Window.partitionBy("user_id", "country").orderBy("ts_unix", "txn_id")) == 1
    has_dev = F.col("device_id").isNotNull() & (F.col("device_id") != "")

    df = (df.withColumn("log_amount", F.log1p("amount"))
          .withColumn("amount_ratio", F.when(F.col("n_prior") > 0, F.col("amount") / (F.col("mean_prior") + 1.0)).otherwise(1.0))
          .withColumn("amount_z", F.when(F.col("n_prior") > 2, (F.col("amount") - F.col("mean_prior")) / (F.col("std_prior") + 1.0)).otherwise(0.0))
          .withColumn("log_amt_sum_24h", F.log1p("amt_sum_24h"))
          .withColumn("log_secs_since_prev", F.log1p(F.coalesce("secs_since_prev", F.lit(30 * 86400))))
          .withColumn("is_online", (F.col("channel") == "online").cast("int"))
          .withColumn("foreign", (F.col("country") != F.col("home_country")).cast("int"))
          .withColumn("new_device", (has_dev & first_dev & (F.col("n_prior") > 0)).cast("int"))
          .withColumn("new_country", (first_ctry & (F.col("n_prior") > 0)).cast("int"))
          .withColumn("hour", F.hour("ts")).withColumn("dow", F.dayofweek("ts"))
          .withColumn("is_night", ((F.hour("ts") < 6) | (F.hour("ts") >= 23)).cast("int"))
          .withColumn("label", F.col("is_fraud").cast("double")))
    for c in CATS:
        df = df.withColumn(f"cat_{c}", (F.col("merchant_category") == c).cast("int"))
    return df

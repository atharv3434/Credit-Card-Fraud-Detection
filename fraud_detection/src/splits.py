from datetime import timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config as C


def bounds(days):
    return C.START + timedelta(days=days[0]), C.START + timedelta(days=days[1])


def split(df: DataFrame):
    """Temporal split with a label-maturity gap between sets. Returns (train, val, test)."""
    def part(days):
        lo, hi = bounds(days)
        return df.filter((F.col("ts") >= F.lit(lo)) & (F.col("ts") < F.lit(hi)))
    return part(C.TRAIN_DAYS), part(C.VAL_DAYS), part(C.TEST_DAYS)

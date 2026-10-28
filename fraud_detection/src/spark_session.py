from pyspark.sql import SparkSession


def get_spark(app_name: str = "fraud") -> SparkSession:
    spark = (SparkSession.builder.appName(app_name).master("local[*]")
             .config("spark.sql.shuffle.partitions", "16").config("spark.driver.memory", "2g")
             .config("spark.sql.session.timeZone", "UTC").config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark

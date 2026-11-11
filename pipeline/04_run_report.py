# Databricks notebook source
# MAGIC %md
# MAGIC # Run report
# MAGIC
# MAGIC What the last pipeline runs produced for one city, next to every other city already in the lakehouse.
# MAGIC Read-only: it changes nothing, so it can be run after any job run, including a failed one.

# COMMAND ----------

from pyspark.sql import functions as F

import airbnb_etl as etl

dbutils.widgets.text("city", "paris")
city = etl.city_slug(dbutils.widgets.get("city"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 · Ingestion runs

# COMMAND ----------

display(
    spark.table(etl.table("bronze", "ingestion_log"))
    .filter(F.col("city") == city)
    .select("run_id", "pipeline_version", "snapshot_date", "source_file", "file_size_bytes", "rows", "columns", "ingested_at")
    .orderBy(F.desc("ingested_at"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Silver quality, all cities, latest run each

# COMMAND ----------

quality = spark.table(etl.table("silver", "quality_report"))
latest = quality.groupBy("city").agg(F.max("created_at").alias("created_at"))
latest_quality = quality.join(latest, ["city", "created_at"])

display(
    latest_quality.select(
        "city", "pipeline_version", "bronze_rows", "invalid_id_rows", "duplicate_rows", "silver_rows",
        "priced_rows", "priced_share", "price_flags", "bathroom_anomalies",
    ).orderBy("city")
)

display(
    latest_quality.select(
        "city", "missing_optional_columns", "empty_columns", "parse_failures",
        F.size("extra_source_columns").alias("extra_columns")
    ).orderBy("city")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03 · NULL % of key features by city

# COMMAND ----------

display(
    latest_quality.select("city", F.explode("feature_null_pct").alias("feature", "null_pct"))
    .groupBy("feature")
    .pivot("city")
    .agg(F.first("null_pct"))
    .orderBy("feature")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 04 · City level

# COMMAND ----------

display(spark.table(etl.table("gold", "city_summary")).orderBy("city"))

display(
    spark.table(etl.table("gold", "room_type_summary"))
    .groupBy("room_type")
    .pivot("city")
    .agg(F.first("share_pct"))
    .orderBy("room_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 05 · Categories the training cities never had

# COMMAND ----------

features = spark.table(etl.table("gold", "listing_features"))
others = features.filter(F.col("city") != city)
current = features.filter(F.col("city") == city)

for column in ["room_type", "property_type", "bathroom_type"]:
    unseen = current.select(column).distinct().join(others.select(column).distinct(), column, "left_anti")
    print(f"{column}: {unseen.count()} values not present in other cities")
    display(current.join(unseen, column).groupBy(column).count().orderBy(F.desc("count")))

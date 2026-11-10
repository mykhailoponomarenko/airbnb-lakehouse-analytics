# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze · raw listings
# MAGIC
# MAGIC One run loads one city. The file lands as-is: every column stays a string, only column names are normalised and lineage columns are added.
# MAGIC Re-running a city replaces only that city's partition, so the step is idempotent.
# MAGIC
# MAGIC | Parameter | Meaning |
# MAGIC |---|---|
# MAGIC | `city` | city name or slug (`paris`, `lyon`, `Pays Basque`) |
# MAGIC | `snapshot_date` | Inside Airbnb snapshot date; empty → date in the most common `scrape_id`, else the most common `last_scraped` |
# MAGIC | `source_url` | optional link to `listings.csv.gz`, used only if the file is not in the volume |
# MAGIC | `run_id`, `pipeline_version` | lineage, filled by the job |

# COMMAND ----------

import os
from datetime import date, datetime, timezone

from pyspark.sql import functions as F

import airbnb_etl as etl

dbutils.widgets.text("city", "paris")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.text("source_url", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("pipeline_version", "dev")

ctx = etl.run_context(dbutils.widgets)
city = ctx["city"]
print(ctx)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 · Locate the source file

# COMMAND ----------

etl.ensure_layers(spark)

source_url = dbutils.widgets.get("source_url").strip()
source_file = etl.resolve_source(city, source_url)
file_size = os.path.getsize(source_file)
print(f"{city}: {source_file} ({file_size / 1e6:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Land the file in bronze

# COMMAND ----------

raw = etl.standardize_columns(etl.read_listings_csv(spark, source_file))

snapshot_date = dbutils.widgets.get("snapshot_date").strip() or etl.infer_snapshot_date(raw)
if not snapshot_date:
    raise ValueError("No snapshot_date parameter and no parsable last_scraped column")

bronze_table = etl.table("bronze", "listings_raw")
etl.overwrite_city(etl.to_bronze(raw, city, snapshot_date, source_file, ctx["run_id"]), bronze_table, city)

landed = spark.table(bronze_table).filter(F.col("city") == city)
rows = landed.count()
print(f"{city} | snapshot {snapshot_date} | {rows:,} rows | {len(raw.columns)} source columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03 · Ingestion log

# COMMAND ----------

log_table = etl.table("bronze", "ingestion_log")
etl.append_log(
    spark,
    [(
        ctx["run_id"], ctx["pipeline_version"], city, date.fromisoformat(snapshot_date),
        source_file, source_url or None, file_size, rows, len(raw.columns), raw.columns,
        datetime.now(timezone.utc),
    )],
    etl.INGESTION_LOG_SCHEMA,
    log_table,
)

display(spark.table(log_table).filter(F.col("city") == city).orderBy(F.desc("ingested_at")).limit(5))

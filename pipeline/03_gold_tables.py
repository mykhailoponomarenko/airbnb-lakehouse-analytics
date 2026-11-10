# Databricks notebook source
# MAGIC %md
# MAGIC # Gold · metrics and model features
# MAGIC
# MAGIC | Table | Grain | Used by |
# MAGIC |---|---|---|
# MAGIC | `gold.city_summary` | city | dashboard *City overview*, *Cross-city comparison*, Genie |
# MAGIC | `gold.neighbourhood_summary` | city × neighbourhood | dashboard maps and rankings, Genie |
# MAGIC | `gold.room_type_summary` | city × room type | room-type mix, Genie |
# MAGIC | `gold.listing_features` | listing | price model (train/test split included), Genie |
# MAGIC
# MAGIC `gold.price_predictions` is written by the model notebook, not here.
# MAGIC
# MAGIC `split` is fixed by a hash of `(city, listing_id)`: 20% test per city, the same on every run, so the test set is chosen before any model and never moves.
# MAGIC `dist_to_center_km` is measured to the median coordinates of the city's own listings, so no city centre is hard-coded.

# COMMAND ----------

from pyspark.sql import functions as F

import airbnb_etl as etl

dbutils.widgets.text("city", "paris")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("pipeline_version", "dev")

ctx = etl.run_context(dbutils.widgets)
city = ctx["city"]
print(ctx)

silver = spark.table(etl.table("silver", "listings")).filter(F.col("city") == city)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 · Metrics

# COMMAND ----------

builders = {
    "city_summary": etl.city_summary,
    "neighbourhood_summary": etl.neighbourhood_summary,
    "room_type_summary": etl.room_type_summary,
    "listing_features": etl.listing_features,
}

for name, build in builders.items():
    etl.overwrite_city(build(silver), etl.table("gold", name), city)
    print(f"gold.{name}: {spark.table(etl.table('gold', name)).filter(F.col('city') == city).count():,} rows for {city}")

display(spark.table(etl.table("gold", "city_summary")).orderBy("city"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Model features

# COMMAND ----------

features = spark.table(etl.table("gold", "listing_features")).filter(F.col("city") == city)

display(
    features.groupBy("split", "price_flag")
    .count()
    .orderBy("split", "price_flag")
)

display(
    features.select(
        [F.round(100 * F.avg(F.col(c).isNull().cast("int")), 2).alias(c) for c in etl.FEATURE_COLUMNS]
    )
)

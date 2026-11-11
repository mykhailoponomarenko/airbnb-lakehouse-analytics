# Databricks notebook source
# MAGIC %md
# MAGIC # Airbnb Exploratory Data Analysis
# MAGIC ## Paris (A) & Lyon (B)
# MAGIC
# MAGIC Region France, currency EUR. Bordeaux (city C) is **not** loaded here; it stayed unseen until the week-1 pipeline and model were frozen.
# MAGIC Numbers quoted in the text come from running this notebook on the snapshots below.
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC **What we know**
# MAGIC * One row = one listing at one scrape. Paris: 77,149 listings, Lyon: 9,201. Both have the same 90 columns and no duplicate IDs.
# MAGIC * **44% of listings have no price** (Paris 43.6%, Lyon 44.9%). The gaps are not random. 97–98% of listings with 0 available days in the next year have no price, against 8–24% of the rest. 75–81% of listings whose last review is over a year old have no price. A missing price marks an inactive listing, so these rows cannot be used for training or evaluation and must not be imputed.
# MAGIC * Prices are heavily right-skewed. Paris median €218 (IQR €139–359), Lyon €110 (€79–158), so **Paris is twice as expensive**. 7–8% of priced listings sit above the IQR fence and are real luxury stock. 345 Paris and 9 Lyon listings cost more than 10× the city median (max €100,666). Several of those are placeholders, e.g. a listing named "indisponible" (French for "unavailable") at €34,537.
# MAGIC * Review scores are missing exactly when `number_of_reviews = 0` (100% vs 0%). That is structural, not a data error.
# MAGIC * Availability is bimodal: 34% (Paris) / 40% (Lyon) of listings have 0 open days, and 22–24% are open more than 270 days.
# MAGIC * **Leakage:** `price_quote_price_per_night` equals `price` in 100% of rows, and `estimated_revenue_l365d` equals `price × estimated_occupancy_l365d` in 100% of rows. Both, and every other `price_quote_*` column, are excluded.
# MAGIC * 14 source columns are empty in every row of both cities, including `host_since`, `instant_bookable`, `bathrooms` and `neighbourhood_group_cleansed`. *(Found in week 2: the week-1 version of this notebook missed it, and two model features were silently empty. See `docs/city_c.md`, change P1.)*
# MAGIC
# MAGIC **What we still do not know**
# MAGIC * Why a listing with open days still has no price (8–24% of them). It might be a failed price quote at scrape time or a minimum stay longer than the quote window; the data does not say.
# MAGIC * Which extreme prices are real (event venues, 12-bedroom houses) and which are placeholders. We flag them; we cannot verify them.
# MAGIC * Whether the advertised price is the price people pay. There is no booking data, and the calendar file is not used.
# MAGIC * How well anything learned on Paris and Lyon transfers to a third city with a different price level and housing stock. That is what the city C run tests.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 · Data sources and setup
# MAGIC
# MAGIC | City | Role | Snapshot (Inside Airbnb) | File | Listings | Columns |
# MAGIC |---|---|---|---|---:|---:|
# MAGIC | Paris | A | 2026-09-16 | `listings.csv.gz` (detailed) | 77,149 | 90 |
# MAGIC | Lyon | B | 2026-09-17 | `listings.csv.gz` (detailed) | 9,201 | 90 |
# MAGIC
# MAGIC **Grain:** one row per listing per snapshot (`id`). `last_scraped` falls 1–5 days after the snapshot date because the scrape runs over several days.
# MAGIC Files are read from the bronze volume exactly as downloaded (`multiLine`, because descriptions contain line breaks).

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

VOLUME = "/Volumes/airbnb_lakehouse/bronze/raw_files"
CITY_PATHS = {
    "Paris": f"{VOLUME}/paris_listings.csv.gz",
    "Lyon": f"{VOLUME}/lyon_listings.csv.gz",
}
SNAPSHOT_DATES = {"Paris": "2026-09-16", "Lyon": "2026-09-17"}


def load_listings(path):
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )

raw = {city: load_listings(path) for city, path in CITY_PATHS.items()}
paris, lyon = raw["Paris"], raw["Lyon"]

for city, df in raw.items():
    print(f"{city} | snapshot: {SNAPSHOT_DATES[city]} | rows: {df.count():,} | columns: {len(df.columns)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Dataset integrity and schema
# MAGIC

# COMMAND ----------

paris_cols, lyon_cols = set(paris.columns), set(lyon.columns)
print("Shared columns:", len(paris_cols & lyon_cols))
print("Only in Paris:", sorted(paris_cols - lyon_cols))
print("Only in Lyon:", sorted(lyon_cols - paris_cols))

key_columns = [
    "id", "price", "room_type", "accommodates", "bedrooms", "beds",
    "bathrooms_text", "neighbourhood_cleansed", "availability_365",
    "number_of_reviews", "review_scores_rating"
]

raw_types = [
    (city, column, dict(df.dtypes).get(column, "missing"))
    for city, df in raw.items()
    for column in key_columns
]
display(spark.createDataFrame(raw_types, ["city", "column", "raw_type"]).orderBy("column", "city"))

for city, df in raw.items():
    print(f"{city} — source data preview")
    display(df.select(*key_columns).limit(5))


# COMMAND ----------

for city, df in raw.items():
    total = df.count()
    unique_ids = df.select("id").distinct().count()
    print(f"{city}: {total:,} rows | {unique_ids:,} distinct IDs | {total - unique_ids:,} duplicate IDs")


# COMMAND ----------

# MAGIC %md
# MAGIC Both cities share all 90 columns and every raw column is a string, so every numeric field has to be parsed. `id` is unique in both files.
# MAGIC
# MAGIC ## 03 · Missing values

# COMMAND ----------

def missing_report(df, city):
    n = df.count()
    expressions = [
        F.sum(
            F.when(
                F.col(name).isNull() | (F.trim(F.col(name)) == ""), 1
            ).otherwise(0)
        ).alias(name)
        for name in df.columns
    ]
    counts = df.agg(*expressions).first().asDict()
    rows = [
        (city, name, int(counts[name]), round(100 * counts[name] / n, 2))
        for name in df.columns
    ]
    return spark.createDataFrame(rows, ["city", "column", "missing_count", "missing_pct"])

missing_summary = missing_report(paris, "Paris").unionByName(missing_report(lyon, "Lyon"))
missing_comparison = (
    missing_summary.groupBy("column")
    .pivot("city", ["Paris", "Lyon"])
    .agg(F.first("missing_pct"))
    .orderBy(F.desc("Paris"), "column")
)

display(missing_comparison)


# COMMAND ----------

# MAGIC %md
# MAGIC **How to read the missing values.** The table above has three groups.
# MAGIC * **Empty in every row** (`host_since`, `instant_bookable`, `bathrooms`, `host_response_*`, `neighbourhood`, `neighbourhood_group_cleansed`, `calendar_updated`, …). Inside Airbnb no longer fills them. `host_since` was replaced by `hosts_time_as_host_years`, and `bathrooms_text` replaces `bathrooms`. They must never be relied on (cell below).
# MAGIC * **Structurally missing**: `review_scores_*`, `first_review` and `reviews_per_month` (18–20%) are NULL exactly when the listing has no reviews. A model can use "no reviews yet" as information; filling in a score would invent one.
# MAGIC * **Missing because the listing is inactive**: `price`, `price_quote_*` and `estimated_revenue_l365d` (44–45%) are missing together. Section 05 shows the pattern.
# MAGIC * Real gaps are small: `beds` 1.5–3.2% (mostly hotel and private rooms), `minimum_nights` 0.2–0.7%, `bathrooms_text` 0.1%.

# COMMAND ----------

empty_everywhere = (
    missing_summary.groupBy("column").agg(F.min("missing_pct").alias("min_missing_pct"))
    .filter("min_missing_pct = 100").orderBy("column")
)
display(empty_everywhere)

# COMMAND ----------

FEATURES = [
    "room_type", "accommodates", "bedrooms", "beds", "bathrooms_text",
    "neighbourhood_cleansed", "latitude", "longitude", "minimum_nights"
]

focus_columns = FEATURES + ["price", "availability_365", "review_scores_rating"]
display(
    missing_comparison
    .filter(F.col("column").isin(focus_columns))
    .orderBy("column")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 04 · Reusable typed features
# MAGIC
# MAGIC The same parsing rules are implemented in the pipeline (`pipeline/airbnb_etl.py`, silver). Prices look like `€1,234.00`. `bathrooms_text` such as "1.5 shared baths" or "Half-bath" is the only usable bathroom source.

# COMMAND ----------

def parse_price(column):
    return F.expr(f"try_cast(regexp_replace(`{column}`, '[$€,]', '') AS DOUBLE)")


def prepare_listings(df, city):
    text = F.lower(F.trim(F.col("bathrooms_text")))
    number = F.regexp_extract(text, r"(\d+(?:\.\d+)?)", 1)

    return (
        df.withColumn("city", F.lit(city))
        .withColumn("price_numeric", parse_price("price"))
        .withColumn("quoted_price_numeric", parse_price("price_quote_price_per_night"))
        .withColumn(
            "has_valid_price",
            F.col("price_numeric").isNotNull() & (F.col("price_numeric") > 0),
        )
        .withColumn("availability_numeric", F.expr("try_cast(availability_365 AS INT)"))
        .withColumn(
            "bathrooms_numeric",
            F.when(text.rlike(r"half[- ]bath"), F.lit(0.5))
            .when(number != "", number.cast("double"))
            .otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "bathroom_type",
            F.when(text.isNull(), "Unknown")
            .when(text.contains("shared"), "Shared")
            .when(text.contains("private"), "Private")
            .otherwise("Standard"),
        )
        .withColumn(
            "bathroom_anomaly",
            (F.col("bathrooms_numeric") > 8)
            & ((F.col("room_type") == "Private room") | (F.expr("try_cast(accommodates AS INT)") <= 5)),
        )
    )

prepared = {city: prepare_listings(df, city) for city, df in raw.items()}
paris_eda, lyon_eda = prepared["Paris"], prepared["Lyon"]
combined_eda = paris_eda.unionByName(lyon_eda)

print("Derived EDA column types:")
for name in ["price_numeric", "has_valid_price", "availability_numeric", "bathrooms_numeric", "bathroom_type"]:
    print(f"  {name}: {dict(paris_eda.dtypes)[name]}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 05 · Price coverage and missingness patterns
# MAGIC

# COMMAND ----------

price_coverage = (
    combined_eda.groupBy("city")
    .agg(
        F.count("*").alias("total_listings"),
        F.sum(F.col("has_valid_price").cast("int")).alias("valid_prices"),
    )
    .withColumn("missing_or_invalid", F.col("total_listings") - F.col("valid_prices"))
    .withColumn("valid_pct", F.round(100 * F.col("valid_prices") / F.col("total_listings"), 2))
    .orderBy("city")
)

display(price_coverage)

display(
    combined_eda.groupBy("city", "room_type")
    .agg(
        F.count("*").alias("listings"),
        F.sum((~F.col("has_valid_price")).cast("int")).alias("missing_prices"),
    )
    .withColumn("missing_pct", F.round(100 * F.col("missing_prices") / F.col("listings"), 2))
    .orderBy("city", F.desc("missing_pct"))
)


# COMMAND ----------

# MAGIC %md
# MAGIC **Price missingness pattern.** About 44% of listings in both cities have no valid price. Room type is not the driver: 40–51% of entire homes and private rooms have no price in both cities. Section 07 shows the driver: listings with **0 available days** have no price in 97–98% of cases. Listings last reviewed more than a year ago miss a price in 75–81% of cases, against 17–24% for recently reviewed ones. The price matches the scraper's booking quote exactly (section 09), and a closed calendar gives no quote. **Missing price therefore means "not bookable right now". The data is missing *not* at random, and it is not an unknown price of an active listing.** Consequences: these rows are excluded from training and from MAE. Predictions for them are "what it would cost if listed". The dashboard reports `priced_pct` next to every median price.
# MAGIC
# MAGIC ## 06 · Price distribution and high-value anomalies

# COMMAND ----------

price_stats = (
    combined_eda.filter(F.col("has_valid_price"))
    .groupBy("city")
    .agg(
        F.count("*").alias("listings"),
        F.round(F.avg("price_numeric"), 2).alias("mean"),
        F.round(F.min("price_numeric"), 2).alias("minimum"),
        F.round(F.percentile_approx("price_numeric", 0.25), 2).alias("q25"),
        F.round(F.percentile_approx("price_numeric", 0.50), 2).alias("median"),
        F.round(F.percentile_approx("price_numeric", 0.75), 2).alias("q75"),
        F.round(F.percentile_approx("price_numeric", 0.95), 2).alias("p95"),
        F.round(F.percentile_approx("price_numeric", 0.99), 2).alias("p99"),
        F.round(F.max("price_numeric"), 2).alias("maximum"),
    )
    .orderBy("city")
)

display(price_stats)

display(
    combined_eda.filter(F.col("has_valid_price"))
    .groupBy("city", "room_type")
    .agg(
        F.count("*").alias("listings"),
        F.round(F.percentile_approx("price_numeric", 0.5), 2).alias("median_price"),
    )
    .orderBy("city", "room_type")
)


# COMMAND ----------

for city, df in prepared.items():
    valid = df.filter(F.col("has_valid_price"))
    median, p99 = valid.approxQuantile("price_numeric", [0.5, 0.99], 0.001)
    prices = (
        valid.filter(F.col("price_numeric") <= p99)
        .select("price_numeric")
        .toPandas()["price_numeric"]
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(prices, bins=55, edgecolor="white", linewidth=0.4)
    ax.axvline(median, linestyle="--", label=f"Median: {median:.2f}")
    ax.set(title=f"{city} | Advertised nightly price", xlabel="Source price units", ylabel="Listings")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# COMMAND ----------

outlier_rows = []

for city, df in prepared.items():
    valid = df.filter(F.col("has_valid_price"))
    q1, median, q3 = valid.approxQuantile("price_numeric", [0.25, 0.5, 0.75], 0.001)
    upper_fence = q3 + 1.5 * (q3 - q1)
    total = valid.count()
    outliers = valid.filter(F.col("price_numeric") > upper_fence).count()
    extreme = valid.filter(F.col("price_numeric") > 10 * median).count()
    outlier_rows.append((city, total, round(upper_fence, 2), outliers, round(100 * outliers / total, 2), extreme))

outlier_summary = spark.createDataFrame(
    outlier_rows,
    ["city", "priced_listings", "iqr_upper_fence", "above_fence", "outlier_pct", "above_10x_median"],
)
display(outlier_summary)


# COMMAND ----------

for city, df in prepared.items():
    print(f"{city} | 10 most expensive listings — review candidates, not confirmed errors")
    display(
        df.filter(F.col("has_valid_price"))
        .select("id", "name", "price_numeric", "room_type", "accommodates", "bedrooms", "neighbourhood_cleansed")
        .orderBy(F.desc("price_numeric"))
        .limit(10)
    )


# COMMAND ----------

# MAGIC %md
# MAGIC **Outliers and how we classify them.** Paris: Q1 €139, median €218, Q3 €359, IQR fence €689. 3,371 listings (7.7%) are above the fence, 345 are above 10× the median, and 4 are below 1/10 of it. Lyon: fence €277, 381 above it (7.5%), 9 above 10× the median.
# MAGIC * **high** (above the fence, below 10× the median): large or premium listings. They are plausible, so we keep them.
# MAGIC * **extreme** (more than 10× or less than 1/10 of the city median): a mix of real luxury (a 12-bedroom house in Lyon 3e, an event loft) and obvious placeholders ("indisponible" at €34,537; a 2-guest studio at €46,789). We cannot tell them apart reliably from the columns, so the pipeline **flags** them (`price_flag`) instead of deleting them. The model uses a log-price target in v2 so they cannot dominate the fit. MAE is reported on all priced listings, extremes included.
# MAGIC * Price is not comparable across cities without context: the same room type costs about 2× more in Paris (entire home median €226 vs €120). A pooled model will learn Paris prices.
# MAGIC
# MAGIC ## 07 · Calendar availability

# COMMAND ----------

availability_stats = (
    combined_eda.groupBy("city")
    .agg(
        F.count("*").alias("listings"),
        F.round(F.avg("availability_numeric"), 2).alias("mean_days"),
        F.percentile_approx("availability_numeric", 0.5).alias("median_days"),
        F.min("availability_numeric").alias("minimum"),
        F.max("availability_numeric").alias("maximum"),
        F.sum(
            (F.col("availability_numeric").isNull() | ~F.col("availability_numeric").between(0, 365)).cast("int")
        ).alias("invalid_availability"),
    )
    .orderBy("city")
)
display(availability_stats)

availability_df = (
    combined_eda.withColumn(
        "availability_group",
        F.when(F.col("availability_numeric") == 0, "0 days")
        .when(F.col("availability_numeric").between(1, 90), "1–90 days")
        .when(F.col("availability_numeric").between(91, 180), "91–180 days")
        .when(F.col("availability_numeric").between(181, 270), "181–270 days")
        .when(F.col("availability_numeric").between(271, 365), "271–365 days")
        .otherwise("Invalid / missing"),
    )
)

availability_distribution = (
    availability_df.groupBy("city", "availability_group")
    .agg(F.count("*").alias("listings"))
    .withColumn("total_city", F.sum("listings").over(Window.partitionBy("city")))
    .withColumn("percentage", F.round(100 * F.col("listings") / F.col("total_city"), 2))
    .orderBy("city", "availability_group")
)
display(availability_distribution)


# COMMAND ----------

buckets = ["0 days", "1–90 days", "91–180 days", "181–270 days", "271–365 days"]
chart_data = availability_distribution.select("city", "availability_group", "percentage").toPandas()

for city in CITY_PATHS:
    city_values = chart_data[chart_data["city"] == city].set_index("availability_group")
    values = [float(city_values.loc[b, "percentage"]) for b in buckets]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(buckets, values)
    ax.set(title=f"{city} | Availability in the next 365 days", xlabel="Available-day range", ylabel="Share of listings (%)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# COMMAND ----------

price_missing_by_availability = (
    availability_df.groupBy("city", "availability_group")
    .agg(
        F.count("*").alias("listings"),
        F.sum((~F.col("has_valid_price")).cast("int")).alias("missing_prices"),
    )
    .withColumn("missing_price_pct", F.round(100 * F.col("missing_prices") / F.col("listings"), 2))
    .orderBy("city", "availability_group")
)
display(price_missing_by_availability)


# COMMAND ----------

# MAGIC %md
# MAGIC **Availability.** It is bimodal. 34% (Paris) / 40% (Lyon) of listings have 0 open days and 22–24% are open 271–365 days; the medians are 83 and 65 days. A 0 means "blocked or booked", not "fully booked", so availability is not a demand signal on its own. Availability explains whether a price exists far better than room type does (table above).
# MAGIC
# MAGIC ## 08 · Feature readiness and bathroom engineering

# COMMAND ----------

feature_quality = (
    missing_summary.filter(F.col("column").isin(FEATURES))
    .groupBy("column")
    .pivot("city", ["Paris", "Lyon"])
    .agg(F.first("missing_pct"))
    .orderBy("column")
)
display(feature_quality)

numeric_rules = {"accommodates": 1, "bedrooms": 0, "beds": 0, "minimum_nights": 1}
numeric_rows = []

for city, df in prepared.items():
    priced = df.filter(F.col("has_valid_price"))
    for name, minimum in numeric_rules.items():
        number = F.expr(f"try_cast(`{name}` AS DOUBLE)")
        row = priced.agg(
            F.count("*").alias("total"),
            F.sum((number.isNull() | (number < minimum)).cast("int")).alias("invalid"),
            F.min(number).alias("min_value"),
            F.max(number).alias("max_value"),
        ).first()
        numeric_rows.append((city, name, row["total"], row["invalid"], row["min_value"], row["max_value"]))

numeric_quality = spark.createDataFrame(
    numeric_rows, ["city", "feature", "priced_listings", "missing_or_invalid", "min_value", "max_value"]
)
display(numeric_quality)


# COMMAND ----------

bathroom_quality = (
    combined_eda.groupBy("city")
    .agg(
        F.count("*").alias("listings"),
        F.sum(F.col("bathrooms_numeric").isNull().cast("int")).alias("unparsed"),
        F.sum((F.col("bathrooms_numeric") == 0).cast("int")).alias("reported_zero"),
        F.sum(F.col("bathroom_anomaly").cast("int")).alias("flagged_for_review"),
    )
    .withColumn("unparsed_pct", F.round(100 * F.col("unparsed") / F.col("listings"), 2))
    .orderBy("city")
)
display(bathroom_quality)

for city, df in prepared.items():
    print(f"{city} | unusual bathroom records (sample)")
    display(
        df.filter(F.col("bathroom_anomaly"))
        .select("id", "name", "room_type", "accommodates", "bedrooms", "bathrooms_text", "bathrooms_numeric")
        .orderBy(F.desc("bathrooms_numeric"))
        .limit(10)
    )


# COMMAND ----------

# MAGIC %md
# MAGIC **Feature readiness.** `accommodates`, `room_type`, coordinates and neighbourhood are complete. `beds` is missing for 1.5–3.2% of listings (mostly hotel rooms), so it is imputed with the training median and gets an indicator. The numeric `bathrooms` column is empty, so the count is parsed from `bathrooms_text`, and "shared/private" becomes `bathroom_type`. 9 Paris and 1 Lyon listings report more than 8 bathrooms while being a private room or sleeping 5 or fewer. These are set to NULL and flagged. `minimum_nights` ranges up to over 365 (14 Paris listings), and about 20% of listings require 30+ nights, i.e. they are mid-term rentals. They are kept, because minimum stay is a legitimate price driver.
# MAGIC
# MAGIC ## Cleaning and transformations the pipeline must apply
# MAGIC
# MAGIC | # | Rule (silver unless noted) | Reason (finding above) |
# MAGIC |---|---|---|
# MAGIC | 1 | Use `try_cast` for every column; booleans `t/f`, percentages `95%`, prices `€1,234.00`, amenities JSON → array | all raw columns are strings; one bad value must not crash a run |
# MAGIC | 2 | Drop rows without a numeric `id`; on a duplicate id, keep the latest `last_scraped` | grain = one listing; no duplicates today, but the pipeline must not depend on that |
# MAGIC | 3 | Price ≤ 0 → NULL; `has_price` flag; never impute price | missing price = inactive listing (MNAR, section 05) |
# MAGIC | 4 | `price_flag` = missing / normal / high (above the city IQR fence) / extreme (> 10× or < 1/10 of the city median) | outliers are classified, not deleted (section 06) |
# MAGIC | 5 | Out-of-range → NULL: availability outside 0–365, scores outside 0–5, accommodates < 1, minimum nights < 1 | impossible values are missing values |
# MAGIC | 6 | Bathrooms from `bathrooms_text` (half-bath = 0.5), `bathroom_type`, more than 8 in a small or private listing → NULL + flag | the numeric column is empty; data-entry errors (section 08) |
# MAGIC | 7 | `amenities_count` from the JSON array | the raw field is a JSON string |
# MAGIC | 8 | `host_years` from `host_since`, falling back to `hosts_time_as_host_years` *(week-2 fix)* | `host_since` is empty in 2026 snapshots |
# MAGIC | 9 | Report contract columns that are NULL in every row (`empty_columns`) *(week-2 fix)* | a present-but-empty column passed every week-1 check |
# MAGIC | 10 | Gold: `dist_to_center_km` to the city's own median listing point; a fixed 20% test split by hash of (city, id) | no hard-coded city centre; the test set is chosen before any model |
# MAGIC | 11 | Leakage columns stay in silver for analysis and never reach `gold.listing_features` | section 09 |
# MAGIC
# MAGIC ## 09 · Target leakage review

# COMMAND ----------

LEAKAGE_COLUMNS = [
    "price", "price_quote_checkin_date", "price_quote_checkout_date",
    "price_quote_total_price", "price_quote_price_per_night", "price_quote_raw",
    "estimated_revenue_l365d",
]

leakage_missingness = (
    missing_summary.filter(F.col("column").isin(LEAKAGE_COLUMNS))
    .groupBy("column")
    .pivot("city", ["Paris", "Lyon"])
    .agg(F.first("missing_pct"))
    .orderBy("column")
)
display(leakage_missingness)

leakage_check = (
    combined_eda.filter(
        F.col("price_numeric").isNotNull() & F.col("quoted_price_numeric").isNotNull()
    )
    .withColumn("absolute_difference", F.abs(F.col("price_numeric") - F.col("quoted_price_numeric")))
    .groupBy("city")
    .agg(
        F.count("*").alias("compared_listings"),
        F.round(100 * F.avg((F.col("absolute_difference") <= 0.01).cast("int")), 2).alias("exact_match_pct"),
        F.round(F.avg("absolute_difference"), 2).alias("mean_absolute_difference"),
    )
    .orderBy("city")
)
display(leakage_check)

revenue_check = (
    combined_eda.withColumn("revenue", F.expr("try_cast(estimated_revenue_l365d AS DOUBLE)"))
    .withColumn("occupancy", F.expr("try_cast(estimated_occupancy_l365d AS DOUBLE)"))
    .filter("revenue IS NOT NULL AND occupancy > 0 AND price_numeric IS NOT NULL")
    .groupBy("city")
    .agg(
        F.count("*").alias("compared_listings"),
        F.round(100 * F.avg((F.abs(F.col("revenue") - F.col("price_numeric") * F.col("occupancy")) <= 1).cast("int")), 2)
        .alias("revenue_equals_price_x_occupancy_pct"),
    )
)
display(revenue_check)

print("Leakage-related columns excluded from model inputs:", LEAKAGE_COLUMNS)
print("Candidate features shared by both cities:", all(c in paris.columns and c in lyon.columns for c in FEATURES))

# COMMAND ----------

# MAGIC %md
# MAGIC **Leakage review: findings**
# MAGIC
# MAGIC | Column(s) | Why it leaks | Evidence |
# MAGIC |---|---|---|
# MAGIC | `price_quote_price_per_night` | it *is* the target: the scraper's quote that `price` is copied from | equal to `price` in 100% of 43,006 (Paris) and 5,069 (Lyon) comparable rows |
# MAGIC | `price_quote_total_price`, `price_quote_raw`, `price_quote_checkin_date`, `price_quote_checkout_date` | total = nightly price × nights of the quote; dates exist only when a quote (= price) exists | present exactly when price is present (55–56%) |
# MAGIC | `estimated_revenue_l365d` | computed from the price: price × estimated occupied nights | equal to `price × estimated_occupancy_l365d` in 100% of rows (cell above) |
# MAGIC | `log_price`, `price_flag`, `has_price` (pipeline-derived) | functions of the target | built in silver/gold, explicitly excluded in `model/price_model.py` |
# MAGIC | `gold.neighbourhood_summary.median_price`, `gold.city_summary.median_price` | aggregates that include the test listings' own prices; joining them as features is target encoding with test leakage | the v2 city price level is computed from **training** rows only, and for a new city only from its calibration split |
# MAGIC
# MAGIC Not leakage, but check before use: `estimated_occupancy_l365d` and `number_of_reviews*` come from reviews, not price, so they are allowed. `availability_*` predicts *whether* a price exists (section 07), not its value; it is allowed as a feature, and models are trained and evaluated only on priced rows. Free-text `name`/`description` can mention a price ("80€/nuit") and is not used.

import os
import re
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = "airbnb_lakehouse"
VOLUME = f"/Volumes/{CATALOG}/bronze/raw_files"
LAYERS = ("bronze", "silver", "gold")

NULL_TOKENS = "('', 'n/a', 'na', 'null', 'none', 'nan', '-')"
RESERVED = {"city", "snapshot_date"}

REQUIRED_COLUMNS = [
    "id", "price", "room_type", "accommodates",
    "neighbourhood_cleansed", "latitude", "longitude",
]

SILVER_CONTRACT = {
    "id": "long",
    "listing_url": "string",
    "name": "string",
    "last_scraped": "date",
    "host_id": "long",
    "host_since": "date",
    "hosts_time_as_host_years": "double",
    "host_response_time": "string",
    "host_response_rate": "percent",
    "host_acceptance_rate": "percent",
    "host_is_superhost": "bool",
    "host_listings_count": "int",
    "host_identity_verified": "bool",
    "neighbourhood_cleansed": "string",
    "neighbourhood_group_cleansed": "string",
    "latitude": "double",
    "longitude": "double",
    "property_type": "string",
    "room_type": "string",
    "accommodates": "int",
    "bathrooms": "double",
    "bathrooms_text": "string",
    "bedrooms": "int",
    "beds": "int",
    "amenities": "json_array",
    "price": "price",
    "minimum_nights": "int",
    "maximum_nights": "int",
    "has_availability": "bool",
    "availability_30": "int",
    "availability_60": "int",
    "availability_90": "int",
    "availability_365": "int",
    "number_of_reviews": "int",
    "number_of_reviews_ltm": "int",
    "number_of_reviews_l30d": "int",
    "first_review": "date",
    "last_review": "date",
    "review_scores_rating": "double",
    "review_scores_accuracy": "double",
    "review_scores_cleanliness": "double",
    "review_scores_checkin": "double",
    "review_scores_communication": "double",
    "review_scores_location": "double",
    "review_scores_value": "double",
    "reviews_per_month": "double",
    "license": "string",
    "instant_bookable": "bool",
    "calculated_host_listings_count": "int",
    "estimated_occupancy_l365d": "int",
    "estimated_revenue_l365d": "double",
    "price_quote_price_per_night": "price",
}

LEAKAGE_COLUMNS = [
    "estimated_revenue_l365d",
    "price_quote_price_per_night",
    "price_quote_total_price",
    "price_quote_raw",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
]

KEY_FEATURES = [
    "room_type", "accommodates", "bedrooms", "beds", "bathrooms",
    "neighbourhood_cleansed", "latitude", "longitude", "minimum_nights",
    "availability_365", "review_scores_rating", "review_scores_location", "amenities",
    "number_of_reviews", "host_since", "hosts_time_as_host_years", "host_is_superhost",
    "calculated_host_listings_count", "instant_bookable",
]

FEATURE_COLUMNS = [
    "room_type", "property_type", "neighbourhood", "accommodates", "bedrooms",
    "beds", "bathrooms", "bathroom_type", "amenities_count", "minimum_nights",
    "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "review_scores_rating", "review_scores_location", "host_is_superhost",
    "host_listings_count", "host_years", "instant_bookable", "latitude",
    "longitude", "dist_to_center_km",
]

TEST_SHARE = 20


def table(layer, name):
    return f"{CATALOG}.{layer}.{name}" if CATALOG else f"{layer}.{name}"


def ensure_layers(spark):
    for layer in LAYERS:
        name = f"{CATALOG}.{layer}" if CATALOG else layer
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {name}")
    if CATALOG:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.bronze.raw_files")


def city_slug(value):
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        raise ValueError("The city parameter is empty")
    return slug


def city_label(slug):
    return slug.replace("_", " ").title()


def run_context(widgets):
    return {
        "city": city_slug(widgets.get("city")),
        "run_id": widgets.get("run_id") or "manual-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "pipeline_version": widgets.get("pipeline_version") or "dev",
    }


# ---------- bronze ----------

def source_candidates(city):
    return [f"{VOLUME}/{city}_listings.csv.gz", f"{VOLUME}/{city}/listings.csv.gz"]


def _is_gzip(path):
    with open(path, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def resolve_source(city, source_url="", retries=3, timeout=120):
    for path in source_candidates(city):
        if os.path.exists(path):
            if path.endswith(".gz") and not _is_gzip(path):
                raise ValueError(f"{path} is not a gzip file; re-download listings.csv.gz for {city}")
            return path

    expected = source_candidates(city)[0]
    if not source_url:
        raise FileNotFoundError(
            f"No listings file for '{city}'. Upload listings.csv.gz to {expected} "
            f"or pass source_url with the Inside Airbnb link."
        )

    errors = []
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(expected, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if not _is_gzip(expected):
                raise ValueError("downloaded file is not gzip (probably an HTML error page)")
            return expected
        except Exception as error:
            errors.append(f"attempt {attempt}: {error}")
            if os.path.exists(expected):
                os.remove(expected)
            time.sleep(3 * attempt)

    raise RuntimeError(
        f"Download failed for '{city}' ({source_url}). Free Edition limits outbound internet, "
        f"so upload the file manually to {expected}. " + " | ".join(errors)
    )


def read_listings_csv(spark, path):
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "PERMISSIVE")
        .csv(path)
    )


def clean_column_name(name):
    name = name.replace("﻿", "").strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_") or "unnamed"
    return f"src_{name}" if name in RESERVED else name


def standardize_columns(df):
    seen, names = {}, []
    for original in df.columns:
        name = clean_column_name(original)
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return df.toDF(*names)


def _most_common(df, expression):
    row = (
        df.select(F.expr(expression).alias("value"))
        .filter(F.col("value").isNotNull())
        .groupBy("value").count()
        .orderBy(F.desc("count"), "value")
        .first()
    )
    return row["value"] if row else None


def infer_snapshot_date(df):
    if "scrape_id" in df.columns:
        stamp = _most_common(df, "substr(trim(scrape_id), 1, 8)")
        try:
            return datetime.strptime(stamp, "%Y%m%d").date().isoformat()
        except (TypeError, ValueError):
            pass
    if "last_scraped" in df.columns:
        value = _most_common(df, "try_cast(trim(last_scraped) AS DATE)")
        return value.isoformat() if value else None
    return None


def to_bronze(raw, city, snapshot_date, source_file, run_id):
    return (
        raw.withColumn("city", F.lit(city))
        .withColumn("snapshot_date", F.lit(snapshot_date).cast("date"))
        .withColumn("_source_file", F.lit(source_file))
        .withColumn("_run_id", F.lit(run_id))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def overwrite_city(df, table_name, city):
    spark = df.sparkSession
    writer = df.write.format("delta").mode("overwrite")
    if spark.catalog.tableExists(table_name):
        writer = writer.option("replaceWhere", f"city = '{city}'").option("mergeSchema", "true")
    else:
        writer = writer.partitionBy("city")
    writer.saveAsTable(table_name)


def append_log(spark, rows, schema, table_name):
    spark.createDataFrame(rows, schema).write.format("delta").mode("append") \
        .option("mergeSchema", "true").saveAsTable(table_name)


INGESTION_LOG_SCHEMA = (
    "run_id string, pipeline_version string, city string, snapshot_date date, "
    "source_file string, source_url string, file_size_bytes bigint, rows bigint, "
    "columns int, column_names array<string>, ingested_at timestamp"
)


# ---------- silver ----------

def _value(column):
    return f"CASE WHEN lower(trim(`{column}`)) IN {NULL_TOKENS} THEN NULL ELSE trim(`{column}`) END"


def cast_sql(column, kind):
    v = _value(column)
    if kind == "string":
        return f"nullif(trim(`{column}`), '')"
    if kind == "int":
        return f"try_cast(try_cast({v} AS DOUBLE) AS INT)"
    if kind == "long":
        return f"try_cast({v} AS BIGINT)"
    if kind == "double":
        return f"try_cast({v} AS DOUBLE)"
    if kind == "date":
        return f"try_cast({v} AS DATE)"
    if kind == "bool":
        return (
            f"CASE WHEN lower({v}) IN ('t', 'true', 'yes', 'y', '1') THEN true "
            f"WHEN lower({v}) IN ('f', 'false', 'no', 'n', '0') THEN false END"
        )
    if kind == "percent":
        return f"try_cast(replace({v}, '%', '') AS DOUBLE) / 100"
    if kind == "price":
        return rf"try_cast(regexp_replace({v}, '[\\p{{Sc}},\\s]', '') AS DOUBLE)"
    if kind == "json_array":
        return f"from_json({v}, 'array<string>')"
    raise ValueError(f"Unknown type '{kind}' for {column}")


def check_contract(source_columns):
    missing_required = [c for c in REQUIRED_COLUMNS if c not in source_columns]
    if missing_required:
        raise ValueError(f"Source is missing required columns: {missing_required}")
    missing_optional = [c for c in SILVER_CONTRACT if c not in source_columns]
    extra = sorted(set(source_columns) - set(SILVER_CONTRACT))
    return missing_optional, extra


def with_contract_columns(df):
    for column in SILVER_CONTRACT:
        if column not in df.columns:
            df = df.withColumn(column, F.lit(None).cast("string"))
    return df


def parse_failures(df):
    expressions = [
        F.sum(
            F.when(F.expr(f"{_value(c)} IS NOT NULL AND ({cast_sql(c, kind)}) IS NULL"), 1).otherwise(0)
        ).alias(c)
        for c, kind in SILVER_CONTRACT.items()
        if kind != "string"
    ]
    counts = df.agg(*expressions).first().asDict()
    return {c: int(n or 0) for c, n in counts.items() if n}


def typed_listings(bronze):
    typed = [F.expr(cast_sql(c, kind)).alias(c) for c, kind in SILVER_CONTRACT.items()]
    meta = [F.col("city"), F.col("snapshot_date"), F.col("_ingested_at").alias("ingested_at")]
    return bronze.select(*meta, *typed).withColumnRenamed("id", "listing_id")


def deduplicate(df):
    latest = Window.partitionBy("city", "listing_id").orderBy(
        F.col("last_scraped").desc_nulls_last(), F.col("number_of_reviews").desc_nulls_last()
    )
    return df.withColumn("_rank", F.row_number().over(latest)).filter("_rank = 1").drop("_rank")


def null_outside(column, low=None, high=None):
    condition = F.lit(True)
    if low is not None:
        condition = condition & (F.col(column) >= low)
    if high is not None:
        condition = condition & (F.col(column) <= high)
    return F.when(condition, F.col(column))


def apply_validity(df):
    rules = {
        "latitude": (-90, 90),
        "longitude": (-180, 180),
        "accommodates": (1, None),
        "bedrooms": (0, None),
        "beds": (0, None),
        "minimum_nights": (1, None),
        "maximum_nights": (1, None),
        "availability_30": (0, 30),
        "availability_60": (0, 60),
        "availability_90": (0, 90),
        "availability_365": (0, 365),
        "number_of_reviews": (0, None),
        "review_scores_rating": (0, 5),
        "review_scores_accuracy": (0, 5),
        "review_scores_cleanliness": (0, 5),
        "review_scores_checkin": (0, 5),
        "review_scores_communication": (0, 5),
        "review_scores_location": (0, 5),
        "review_scores_value": (0, 5),
        "price": (0.01, None),
    }
    for column, (low, high) in rules.items():
        df = df.withColumn(column, null_outside(column, low, high))
    return df


def add_bathrooms(df):
    text = F.lower(F.trim(F.col("bathrooms_text")))
    number = F.regexp_extract(text, r"(\d+(?:\.\d+)?)", 1)
    from_text = (
        F.when(text.rlike(r"half[- ]bath"), F.lit(0.5))
        .when(number != "", number.cast("double"))
    )
    raw = F.coalesce(F.col("bathrooms"), from_text)
    anomaly = (raw > 8) & ((F.col("room_type") == "Private room") | (F.col("accommodates") <= 5))
    return (
        df.withColumn("bathroom_anomaly", F.coalesce(anomaly, F.lit(False)))
        .withColumn("bathrooms", F.when(~F.col("bathroom_anomaly"), raw))
        .withColumn(
            "bathroom_type",
            F.when(text.isNull(), "Unknown")
            .when(text.contains("shared"), "Shared")
            .when(text.contains("private"), "Private")
            .otherwise("Standard"),
        )
    )


def add_price_flags(df):
    stats = (
        df.filter(F.col("price").isNotNull())
        .agg(F.expr("percentile(price, array(0.25, 0.5, 0.75)) AS q"))
        .first()["q"]
    )
    df = df.withColumn("has_price", F.col("price").isNotNull())
    if not stats:
        return df.withColumn("price_flag", F.when(F.col("price").isNull(), "missing"))
    q1, median, q3 = stats
    fence = q3 + 1.5 * (q3 - q1)
    return df.withColumn(
        "price_flag",
        F.when(F.col("price").isNull(), "missing")
        .when((F.col("price") > 10 * median) | (F.col("price") < median / 10), "extreme")
        .when(F.col("price") > fence, "high")
        .otherwise("normal"),
    )


def to_silver(bronze):
    df = typed_listings(bronze)
    df = df.filter(F.col("listing_id").isNotNull())
    df = deduplicate(df)
    df = apply_validity(df)
    df = add_bathrooms(df)
    df = df.withColumn("amenities_count", F.when(F.col("amenities").isNotNull(), F.size("amenities")))
    df = add_price_flags(df)
    return df.withColumn("city_name", F.initcap(F.regexp_replace("city", "_", " ")))


QUALITY_SCHEMA = (
    "run_id string, pipeline_version string, city string, snapshot_date date, "
    "bronze_rows bigint, invalid_id_rows bigint, duplicate_rows bigint, silver_rows bigint, "
    "priced_rows bigint, priced_share double, price_flags map<string,bigint>, "
    "missing_optional_columns array<string>, extra_source_columns array<string>, "
    "parse_failures map<string,bigint>, feature_null_pct map<string,double>, "
    "bathroom_anomalies bigint, created_at timestamp, empty_columns array<string>"
)


def null_pct(df, columns):
    total = df.count()
    if total == 0:
        return {c: None for c in columns}
    row = df.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns]).first()
    return {c: round(100.0 * row[c] / total, 2) for c in columns}


def empty_columns(df):
    """Contract columns that are NULL in every row: present in the file, but carrying nothing."""
    total = df.count()
    if total == 0:
        return []
    columns = [c for c in SILVER_CONTRACT if c != "id" and c in df.columns]
    row = df.agg(*[F.count(F.col(c)).alias(c) for c in columns]).first()
    return [c for c in columns if row[c] == 0]


# ---------- gold ----------

def _median(column):
    return F.expr(f"percentile({column}, 0.5)")


def city_summary(silver):
    priced = F.col("price").isNotNull()
    normal = F.col("price_flag").isin("normal", "high")
    return silver.groupBy("city", "city_name", "snapshot_date").agg(
        F.count("*").alias("listings"),
        F.sum(priced.cast("int")).alias("priced_listings"),
        F.round(F.avg(priced.cast("double")) * 100, 2).alias("priced_pct"),
        F.round(_median("price"), 2).alias("median_price"),
        F.round(F.expr("percentile(price, 0.25)"), 2).alias("p25_price"),
        F.round(F.expr("percentile(price, 0.75)"), 2).alias("p75_price"),
        F.round(F.avg(F.when(normal, F.col("price"))), 2).alias("mean_price_excl_extreme"),
        F.round(_median("availability_365"), 1).alias("median_availability_365"),
        F.round(F.avg("availability_365"), 1).alias("avg_availability_365"),
        F.round(F.avg((F.col("availability_365") == 0).cast("double")) * 100, 2).alias("zero_availability_pct"),
        F.round(F.avg("review_scores_rating"), 3).alias("avg_review_score"),
        F.round(F.avg((F.col("number_of_reviews") > 0).cast("double")) * 100, 2).alias("reviewed_pct"),
        F.countDistinct("host_id").alias("hosts"),
        F.round(F.avg((F.col("calculated_host_listings_count") > 1).cast("double")) * 100, 2).alias("multi_listing_host_pct"),
        F.round(F.avg(F.col("host_is_superhost").cast("double")) * 100, 2).alias("superhost_pct"),
        F.round(F.avg((F.col("room_type") == "Entire home/apt").cast("double")) * 100, 2).alias("entire_home_pct"),
        F.countDistinct("neighbourhood_cleansed").alias("neighbourhoods"),
    ).withColumn("updated_at", F.current_timestamp())


def neighbourhood_summary(silver):
    by_price = Window.partitionBy("city").orderBy(F.desc_nulls_last("median_price"))
    by_size = Window.partitionBy("city").orderBy(F.desc("listings"))
    return (
        silver.groupBy("city", "city_name", F.col("neighbourhood_cleansed").alias("neighbourhood"))
        .agg(
            F.count("*").alias("listings"),
            F.sum(F.col("price").isNotNull().cast("int")).alias("priced_listings"),
            F.round(_median("price"), 2).alias("median_price"),
            F.round(F.avg("review_scores_rating"), 3).alias("avg_review_score"),
            F.round(F.avg("availability_365"), 1).alias("avg_availability_365"),
            F.round(F.avg((F.col("room_type") == "Entire home/apt").cast("double")) * 100, 2).alias("entire_home_pct"),
            F.round(F.avg("latitude"), 5).alias("latitude"),
            F.round(F.avg("longitude"), 5).alias("longitude"),
        )
        .withColumn("price_rank", F.dense_rank().over(by_price))
        .withColumn("size_rank", F.dense_rank().over(by_size))
        .withColumn("updated_at", F.current_timestamp())
    )


def room_type_summary(silver):
    city_total = Window.partitionBy("city")
    return (
        silver.groupBy("city", "city_name", "room_type")
        .agg(
            F.count("*").alias("listings"),
            F.sum(F.col("price").isNotNull().cast("int")).alias("priced_listings"),
            F.round(_median("price"), 2).alias("median_price"),
            _median("accommodates").alias("median_accommodates"),
        )
        .withColumn("share_pct", F.round(100 * F.col("listings") / F.sum("listings").over(city_total), 2))
        .withColumn("updated_at", F.current_timestamp())
    )


def haversine_km(lat1, lon1, lat2, lon2):
    rad = F.lit(3.141592653589793 / 180)
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = F.sin(dlat / 2) ** 2 + F.cos(lat1 * rad) * F.cos(lat2 * rad) * F.sin(dlon / 2) ** 2
    return F.lit(2 * 6371.0) * F.asin(F.sqrt(a))


def listing_features(silver):
    center = silver.agg(
        _median("latitude").alias("center_lat"), _median("longitude").alias("center_lon")
    ).first()
    split_bucket = F.pmod(F.xxhash64("city", "listing_id"), F.lit(100))
    return silver.select(
        "city", "city_name", "listing_id", "snapshot_date", "name", "listing_url",
        F.col("neighbourhood_cleansed").alias("neighbourhood"),
        "room_type", "property_type", "accommodates", "bedrooms", "beds", "bathrooms",
        "bathroom_type", "amenities_count", "minimum_nights", "availability_365",
        "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",
        "review_scores_location", F.col("host_is_superhost").cast("int").alias("host_is_superhost"),
        F.col("calculated_host_listings_count").alias("host_listings_count"),
        # host_since is empty in the 2026 snapshots; Inside Airbnb now ships hosts_time_as_host_years instead
        F.round(
            F.coalesce(F.datediff("snapshot_date", "host_since") / 365.25, F.col("hosts_time_as_host_years")), 2
        ).alias("host_years"),
        F.col("instant_bookable").cast("int").alias("instant_bookable"),
        "latitude", "longitude",
        F.round(
            haversine_km(F.col("latitude"), F.col("longitude"), F.lit(center["center_lat"]), F.lit(center["center_lon"])),
            3,
        ).alias("dist_to_center_km"),
        "price",
        F.round(F.log1p("price"), 6).alias("log_price"),
        "price_flag",
        "has_price",
        F.when(split_bucket < TEST_SHARE, "test").otherwise("train").alias("split"),
        F.current_timestamp().alias("updated_at"),
    )

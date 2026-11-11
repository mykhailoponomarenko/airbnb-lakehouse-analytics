"""Generates dashboards/airbnb_france.lvdash.json (AI/BI dashboard, serialized format).

Edit here, run `python dashboards/make_dashboard.py`, then `databricks bundle deploy`.
"""
import json
import os

G = "airbnb_lakehouse.gold"
datasets, pages = [], []


def dataset(name, sql):
    datasets.append({"name": name, "displayName": name, "queryLines": [l + "\n" for l in sql.strip().split("\n")]})
    return name


def fields(ds, cols):
    return [{"name": "main_query", "query": {"datasetName": ds, "disaggregated": True,
                                             "fields": [{"name": c, "expression": f"`{c}`"} for c in cols]}}]


def frame(title, desc=None):
    f = {"showTitle": True, "title": title}
    if desc:
        f.update(showDescription=True, description=desc)
    return f


def text(name, lines, x, y, w, h):
    # each line needs its own newline, and a blank line separates the title from the subtitle
    nl = chr(10)
    lines = [lines[0] + nl, nl] + [l + nl for l in lines[1:]]
    return {"widget": {"name": name, "multilineTextboxSpec": {"lines": lines}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def bar(name, ds, xcol, ycol, title, x, y, w, h, color=None, xlabel=None, ylabel=None, horizontal=False, desc=None, stack=False):
    cols = [xcol, ycol] + ([color] if color else [])
    xe = {"fieldName": xcol, "scale": {"type": "categorical"}, "displayName": xlabel or xcol}
    ye = {"fieldName": ycol, "scale": {"type": "quantitative"}, "displayName": ylabel or ycol}
    enc = {"x": ye, "y": xe} if horizontal else {"x": xe, "y": ye}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
    enc["label"] = {"show": True}
    spec = {"version": 3, "widgetType": "bar", "encodings": enc, "frame": frame(title, desc)}
    if color:
        spec["mark"] = {"layout": "stack" if stack else "group"}
    return {"widget": {"name": name, "queries": fields(ds, cols), "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def line(name, ds, xcol, ycol, color, title, x, y, w, h, xlabel, ylabel):
    enc = {"x": {"fieldName": xcol, "scale": {"type": "quantitative"}, "displayName": xlabel},
           "y": {"fieldName": ycol, "scale": {"type": "quantitative"}, "displayName": ylabel},
           "color": {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}}
    return {"widget": {"name": name, "queries": fields(ds, [xcol, ycol, color]),
                       "spec": {"version": 3, "widgetType": "line", "encodings": enc, "frame": frame(title)}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def scatter(name, ds, xcol, ycol, color, title, x, y, w, h, desc=None):
    enc = {"x": {"fieldName": xcol, "scale": {"type": "quantitative"}, "displayName": "Actual price (EUR)"},
           "y": {"fieldName": ycol, "scale": {"type": "quantitative"}, "displayName": "Predicted price (EUR)"},
           "color": {"fieldName": color, "scale": {"type": "categorical"}, "displayName": "City"}}
    return {"widget": {"name": name, "queries": fields(ds, [xcol, ycol, color]),
                       "spec": {"version": 3, "widgetType": "scatter", "encodings": enc, "frame": frame(title, desc)}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def table(name, ds, cols, title, x, y, w, h, desc=None):
    enc = {"columns": [{"fieldName": c, "displayName": label} for c, label in cols]}
    return {"widget": {"name": name, "queries": fields(ds, [c for c, _ in cols]),
                       "spec": {"version": 2, "widgetType": "table", "encodings": enc, "frame": frame(title, desc)}},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def page(name, display, layout):
    pages.append({"name": name, "displayName": display, "layout": layout, "pageType": "PAGE_TYPE_CANVAS"})


# ---------- datasets ----------
city = dataset("city_summary", f"""
SELECT city_name, listings, priced_listings, priced_pct, median_price, p25_price, p75_price,
       median_availability_365, zero_availability_pct, avg_review_score, hosts, superhost_pct,
       multi_listing_host_pct, entire_home_pct, neighbourhoods
FROM {G}.city_summary
ORDER BY median_price DESC""")
room = dataset("room_types", f"""
SELECT city_name, room_type, listings, share_pct, median_price
FROM {G}.room_type_summary
ORDER BY city_name, listings DESC""")
nbhd = dataset("top_neighbourhoods", f"""
SELECT city_name, neighbourhood, listings, median_price, avg_review_score
FROM {G}.neighbourhood_summary
WHERE priced_listings >= 20
QUALIFY ROW_NUMBER() OVER (PARTITION BY city ORDER BY median_price DESC) <= 5
ORDER BY CASE city WHEN 'paris' THEN 1 WHEN 'lyon' THEN 2 ELSE 3 END, median_price DESC""")
compare = dataset("paris_vs_lyon", f"""
WITH s AS (SELECT * FROM {G}.city_summary WHERE city IN ('paris', 'lyon')),
long AS (
  SELECT city, metric, value FROM s
  LATERAL VIEW STACK(9,
    '1 Listings', CAST(listings AS DOUBLE),
    '2 Share with a price (%)', CAST(priced_pct AS DOUBLE),
    '3 Median price (EUR)', CAST(median_price AS DOUBLE),
    '4 Entire-home share (%)', CAST(entire_home_pct AS DOUBLE),
    '5 Median available days / year', CAST(median_availability_365 AS DOUBLE),
    '6 Listings with 0 available days (%)', CAST(zero_availability_pct AS DOUBLE),
    '7 Average review score', CAST(avg_review_score AS DOUBLE),
    '8 Multi-listing hosts (%)', CAST(multi_listing_host_pct AS DOUBLE),
    '9 Superhosts (%)', CAST(superhost_pct AS DOUBLE)) t AS metric, value)
SELECT SUBSTRING(metric, 3) AS metric,
       ROUND(MAX(CASE WHEN city = 'paris' THEN value END), 2) AS paris,
       ROUND(MAX(CASE WHEN city = 'lyon' THEN value END), 2) AS lyon,
       ROUND(100 * (MAX(CASE WHEN city = 'lyon' THEN value END) / MAX(CASE WHEN city = 'paris' THEN value END) - 1), 1) AS lyon_vs_paris_pct
FROM long GROUP BY metric ORDER BY metric""")
room_price = dataset("room_price_by_city", f"""
SELECT city_name, room_type, median_price, listings
FROM {G}.room_type_summary
WHERE listings >= 20
ORDER BY room_type, city_name""")
guests = dataset("price_by_guests", f"""
SELECT city_name, LEAST(accommodates, 8) AS guests, PERCENTILE(price, 0.5) AS median_price, COUNT(*) AS listings
FROM {G}.listing_features
WHERE price IS NOT NULL AND room_type = 'Entire home/apt'
GROUP BY ALL ORDER BY city_name, guests""")
metrics = dataset("model_metrics", f"""
SELECT model, INITCAP(city) AS city, variant, labeled_listings, model_mae_eur, baseline_mae_eur, improvement_eur
FROM {G}.v_model_metrics
ORDER BY model, city, variant""")
mae_bar = dataset("mae_by_city_and_model", f"""
SELECT INITCAP(city) AS city,
       CASE WHEN run_id = 'france_v1' THEN 'v1 (week 1)'
            WHEN variant = 'uncalibrated' THEN 'v2 without city prices'
            ELSE 'v2 (week 2)' END AS model_variant,
       model_mae_eur AS mae_eur
FROM {G}.v_model_metrics
WHERE city <> 'combined'
UNION ALL
SELECT INITCAP(city), 'Baseline', baseline_mae_eur
FROM {G}.v_model_metrics
WHERE city <> 'combined' AND run_id = 'france_v2' AND variant IN ('holdout', 'calibrated')
ORDER BY city, model_variant""")
scatter_ds = dataset("predicted_vs_actual_v2", f"""
SELECT city_name, actual_price_eur, predicted_price_eur
FROM {G}.v_test_predictions
WHERE run_id = 'france_v2' AND actual_price_eur <= 1000""")
err_room = dataset("error_by_room_type", f"""
SELECT model, city_name, room_type, COUNT(*) AS listings,
       ROUND(AVG(absolute_error_eur), 2) AS model_mae_eur,
       ROUND(AVG(baseline_absolute_error_eur), 2) AS baseline_mae_eur,
       ROUND(AVG(residual_eur), 2) AS mean_residual_eur
FROM {G}.v_test_predictions
GROUP BY ALL HAVING COUNT(*) >= 10
ORDER BY city_name, room_type, model""")
err_room_v2 = dataset("error_by_room_type_v2", f"""
SELECT CONCAT(city_name, ' - ', room_type) AS segment, ROUND(AVG(absolute_error_eur), 2) AS model_mae_eur
FROM {G}.v_test_predictions
WHERE run_id = 'france_v2'
GROUP BY city_name, room_type HAVING COUNT(*) >= 10
ORDER BY model_mae_eur DESC""")
err_nbhd = dataset("error_by_neighbourhood_v2", f"""
SELECT city_name, neighbourhood, COUNT(*) AS listings,
       ROUND(AVG(absolute_error_eur), 2) AS model_mae_eur,
       ROUND(AVG(residual_eur), 2) AS mean_residual_eur
FROM {G}.v_test_predictions
WHERE run_id = 'france_v2'
GROUP BY ALL HAVING COUNT(*) >= 15
ORDER BY model_mae_eur DESC LIMIT 15""")
worst = dataset("most_mispriced_v2", f"""
SELECT city_name, error_direction, name, neighbourhood, room_type, accommodates,
       actual_price_eur, predicted_price_eur, residual_eur, listing_url
FROM {G}.v_test_predictions
WHERE run_id = 'france_v2' AND price_flag <> 'extreme'
ORDER BY ABS(residual_eur) DESC LIMIT 30""")

# ---------- page 1: city overview ----------
page("city_overview", "City overview", [
    text("t1", ["## Paris costs about twice as much as Lyon or Bordeaux, and 36–44% of listings in every city show no price",
                "Median advertised nightly price, EUR. Listings without a price are inactive (closed calendar), so medians use priced listings only. Snapshots: Paris 2026-09-16, Lyon and Bordeaux 2026-09-17."], 0, 0, 6, 2),
    table("city_table", city, [("city_name", "City"), ("listings", "Listings"), ("priced_pct", "% with price"),
                               ("median_price", "Median price"), ("p25_price", "P25"), ("p75_price", "P75"),
                               ("median_availability_365", "Median open days"), ("avg_review_score", "Avg review"),
                               ("hosts", "Hosts"), ("superhost_pct", "% superhost"), ("entire_home_pct", "% entire home"),
                               ("neighbourhoods", "Neighbourhoods")], "Per-city summary", 0, 2, 6, 4),
    bar("median_price", city, "city_name", "median_price", "Median nightly price (EUR)", 0, 6, 2, 6, xlabel="City", ylabel="Median price (EUR)"),
    bar("room_mix", room, "city_name", "share_pct", "Room-type mix (% of listings)", 2, 6, 2, 6, color="room_type", xlabel="City", ylabel="% of listings", stack=True),
    bar("availability", city, "city_name", "zero_availability_pct", "Listings with 0 open days in the next year (%)", 4, 6, 2, 6, xlabel="City", ylabel="%"),
    table("nbhd_table", nbhd, [("city_name", "City"), ("neighbourhood", "Neighbourhood"), ("listings", "Listings"),
                               ("median_price", "Median price"), ("avg_review_score", "Avg review")],
          "Most expensive neighbourhoods (at least 20 priced listings)", 0, 12, 6, 7),
])

# ---------- page 2: cross-city comparison ----------
page("cross_city", "Cross-city comparison", [
    text("t2", ["## The same listing costs about twice as much in Paris as in Lyon, for every room type",
                "Lyon is not a smaller Paris: the gap is a price level, not a different mix. That is why a model trained on both cities needs to know which city it is pricing."], 0, 0, 6, 2),
    table("compare_table", compare, [("metric", "Metric"), ("paris", "Paris (A)"), ("lyon", "Lyon (B)"),
                                     ("lyon_vs_paris_pct", "Lyon vs Paris (%)")], "Paris against Lyon", 0, 2, 3, 7),
    bar("room_price", room_price, "room_type", "median_price", "Median price by room type (EUR)", 3, 2, 3, 7, color="city_name", xlabel="Room type", ylabel="Median price (EUR)"),
    line("guests_price", guests, "guests", "median_price", "city_name", "Entire homes: median price by number of guests (8 = 8+)", 0, 9, 6, 6, "Guests", "Median price (EUR)"),
])

# ---------- page 3: model ----------
page("model", "Model", [
    text("t3", ["## Once it knows the city's price level, v2 beats the baseline in every city. The week-1 model overpriced 97% of Bordeaux listings",
                "MAE in EUR on held-out listings only: the Paris and Lyon test split, and Bordeaux (city C). v1 = week-1 model, scored on Bordeaux unchanged. v2 = week-2 model, where *calibrated* uses Bordeaux's median price from its training split and *uncalibrated* uses no Bordeaux prices at all. Baseline = median price of the same room type and guest count in Paris+Lyon training data."], 0, 0, 6, 2),
    bar("mae_bar", mae_bar, "city", "mae_eur", "MAE by city: baseline, v1, v2 (EUR, lower is better). Bordeaux v1 = first run on all priced listings", 0, 2, 3, 7, color="model_variant", xlabel="City", ylabel="MAE (EUR)"),
    table("metrics_table", metrics, [("model", "Model"), ("city", "City"), ("variant", "Variant"), ("labeled_listings", "Listings"),
                                     ("model_mae_eur", "Model MAE"), ("baseline_mae_eur", "Baseline MAE"), ("improvement_eur", "Improvement")],
          "Test-set error, all runs", 3, 2, 3, 7),
    scatter("scatter", scatter_ds, "actual_price_eur", "predicted_price_eur", "city_name", "v2: predicted against actual price (held-out, up to EUR 1,000)", 0, 9, 3, 7),
    bar("err_room", err_room_v2, "segment", "model_mae_eur", "v2: MAE by city and room type (EUR)", 3, 9, 3, 7, xlabel="City - room type", ylabel="MAE (EUR)", horizontal=True),
    table("err_room_table", err_room, [("model", "Model"), ("city_name", "City"), ("room_type", "Room type"), ("listings", "Listings"),
                                       ("model_mae_eur", "Model MAE"), ("baseline_mae_eur", "Baseline MAE"), ("mean_residual_eur", "Mean residual (+ = under-priced)")],
          "Error by room type, v1 against v2", 0, 16, 6, 6),
    table("err_nbhd", err_nbhd, [("city_name", "City"), ("neighbourhood", "Neighbourhood"), ("listings", "Listings"),
                                 ("model_mae_eur", "MAE"), ("mean_residual_eur", "Mean residual (+ = under-priced)")],
          "v2: neighbourhoods with the largest error", 0, 22, 6, 6),
    table("worst", worst, [("city_name", "City"), ("error_direction", "Direction"), ("name", "Listing"), ("neighbourhood", "Neighbourhood"),
                           ("room_type", "Room type"), ("accommodates", "Guests"), ("actual_price_eur", "Actual"),
                           ("predicted_price_eur", "Predicted"), ("residual_eur", "Residual"), ("listing_url", "URL")],
          "v2: the 30 most under- or over-priced listings", 0, 28, 6, 8,
          desc="Excludes extreme advertised prices (more than 10x the city median). Without this filter, all 30 largest errors are such extremes, all under-priced: the model does not follow placeholder or luxury-event prices."),
])

out = {"datasets": datasets, "pages": pages,
       "uiSettings": {"theme": {"widgetHeaderAlignment": "ALIGNMENT_UNSPECIFIED"}}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "airbnb_france.lvdash.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(len(datasets), "datasets", sum(len(p["layout"]) for p in pages), "widgets")

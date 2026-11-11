# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # France price prediction
# MAGIC Run `train_ab` first. After A/B is frozen, run the ETL for city C, then `evaluate_c` with the same run ID.
# MAGIC
# MAGIC * `france_v1` — week-1 model (raw-price forest). Its Bordeaux first run is recorded and must not be rerun.
# MAGIC * `france_v2` — week-2 model (log price relative to the city's price level). For a new city the level is the
# MAGIC   median advertised price of that city's ETL `train` split (the calibration sample); MAE is reported on its
# MAGIC   `test` split only, with and without calibration. `reference_run_id` scores an older frozen model on the same rows.

# COMMAND ----------

import sys, os, json, uuid
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from pyspark.sql import functions as F
sys.path.insert(0, os.path.abspath('../model'))
from price_model import MODEL_VERSION, INPUT_COLUMNS, FEATURES, fit, score, metrics, city_levels, POOLED

dbutils.widgets.dropdown('mode', 'train_ab', ['train_ab', 'evaluate_c'])
dbutils.widgets.text('run_id', '')
dbutils.widgets.text('catalog', 'airbnb_lakehouse')
dbutils.widgets.text('ab_cities', 'paris,lyon')
dbutils.widgets.text('c_city', 'bordeaux')
dbutils.widgets.text('reference_run_id', '')
mode = dbutils.widgets.get('mode')
catalog = dbutils.widgets.get('catalog')
assert catalog.replace('_', '').isalnum(), 'Use a simple catalog identifier'
run_id = dbutils.widgets.get('run_id') or str(uuid.uuid4())
assert all(c.isalnum() or c in '-_' for c in run_id)
ab_cities = [c.strip() for c in dbutils.widgets.get('ab_cities').split(',') if c.strip()]
c_city = dbutils.widgets.get('c_city').strip()
reference_run_id = dbutils.widgets.get('reference_run_id').strip()
gold = f'{catalog}.gold'
spark.sql(f'CREATE VOLUME IF NOT EXISTS {gold}.model_artifacts')
artifacts = f'/Volumes/{catalog}/gold/model_artifacts'
folder = f'{artifacts}/{run_id}'

def load(cities):
    source = spark.table(f'{gold}.listing_features').filter(F.col('city').isin(cities))
    required = ['city', 'listing_id', 'snapshot_date', 'price', 'split', 'neighbourhood'] + INPUT_COLUMNS
    missing = set(required) - set(source.columns)
    assert not missing, f'Missing ETL fields: {missing}'
    frame = source.select(*required).toPandas()
    assert set(frame.city) == set(cities), 'Missing city; run ETL first'
    assert not frame.duplicated(['city', 'listing_id']).any(), 'Duplicate listing keys'
    frame['price'] = pd.to_numeric(frame.price, errors='coerce')
    frame.loc[~np.isfinite(frame.price) | (frame.price <= 0), 'price'] = np.nan
    assert frame.split.isin(['train', 'test']).all()
    return frame

def load_frozen(rid):
    with open(f'{artifacts}/{rid}/manifest.json') as f:
        manifest = json.load(f)
    assert manifest['sklearn_version'] == sklearn.__version__, 'Use the original sklearn version'
    return manifest, *joblib.load(f'{artifacts}/{rid}/frozen_model.joblib')

def publish(frame, table):
    # Append preserves first-run evidence and prior model versions.
    frame = frame.copy()
    frame['run_id'] = run_id
    frame['evaluation_stage'] = mode
    (spark.createDataFrame(frame).withColumn('created_at', F.current_timestamp())
     .write.mode('append').option('mergeSchema', 'true').saveAsTable(f'{gold}.{table}'))

# COMMAND ----------

if mode == 'train_ab':
    assert not os.path.exists(folder), 'Run ID already exists; use a fresh ID'
    data = load(ab_cities)
    train = data[(data.split == 'train') & data.price.notna()]
    test = data[(data.split == 'test') & data.price.notna()]
    assert set(train.city) == set(test.city) == set(ab_cities)
    model, baseline = fit(train)
    predictions = score(data, model, baseline)
    predictions['evaluation_set'] = np.where(predictions.split == 'test', 'ab_test', 'ab_train')
    report = metrics(predictions[predictions.evaluation_set == 'ab_test']).assign(variant='holdout')
    os.makedirs(folder)
    joblib.dump((model, baseline), folder + '/frozen_model.joblib')
    manifest = {'run_id': run_id, 'model_version': MODEL_VERSION,
                'frozen_at': datetime.now(timezone.utc).isoformat(),
                'sklearn_version': sklearn.__version__, 'features': FEATURES,
                'city_price_levels': model['levels'],
                'snapshots': data.groupby('city').snapshot_date.apply(lambda x: sorted(set(map(str, x)))).to_dict(),
                'training_rows': len(train), 'test_rows': len(test), 'currency': 'EUR'}
    with open(folder + '/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    report.to_csv(folder + '/ab_test_metrics.csv', index=False)
else:
    assert os.path.exists(folder + '/manifest.json'), 'Provide the frozen A/B run ID'
    assert not os.path.exists(folder + '/c_first_metrics.csv'), 'First city C run is already recorded'
    manifest, model, baseline = load_frozen(run_id)
    data = load([c_city])
    if manifest.get('model_version', 'v1') == 'v1':
        # Week-1 contract: whole city scored, no calibration.
        predictions = score(data, model, baseline)
        predictions['evaluation_set'] = 'c_unseen'
        report = metrics(predictions).assign(variant='unchanged')
    else:
        calibration = data[(data.split == 'train') & data.price.notna()]
        levels = {k: v for k, v in city_levels(calibration).items() if k != POOLED}
        print('City C price level from', len(calibration), 'calibration listings:', levels)
        predictions = score(data, model, baseline, levels)
        predictions['evaluation_set'] = np.where(predictions.split == 'test', 'c_unseen', 'c_calibration')
        test = predictions[predictions.evaluation_set == 'c_unseen']
        variants = [metrics(test).assign(variant='calibrated'),
                    metrics(score(test, model, baseline)).assign(variant='uncalibrated')]
        if reference_run_id:
            _, ref_model, ref_baseline = load_frozen(reference_run_id)
            variants.append(metrics(score(test, ref_model, ref_baseline)).assign(variant=f'reference_{reference_run_id}'))
        report = pd.concat(variants, ignore_index=True)
        with open(folder + '/c_levels.json', 'w') as f:
            json.dump(levels, f, indent=2)
    report = report[report.city != 'combined']
    assert len(predictions) == len(data) and predictions.predicted_price.notna().all(), 'Every listing needs a prediction'
    report.to_csv(folder + '/c_first_metrics.csv', index=False)
    ab = pd.read_csv(folder + '/ab_test_metrics.csv').query("city == 'combined'").iloc[0]
    print(f'{c_city} minus A/B test MAE (EUR):', report.iloc[0].model_mae - ab.model_mae)
    print(f'{c_city} snapshot(s):', sorted(set(map(str, data.snapshot_date))))

# COMMAND ----------

predictions['model_version'] = manifest.get('model_version', 'v1') if mode == 'evaluate_c' else MODEL_VERSION
publish(predictions[['city', 'listing_id', 'snapshot_date', 'neighbourhood', 'room_type',
                     'accommodates', 'price', 'predicted_price', 'baseline_price', 'residual',
                     'absolute_error', 'baseline_absolute_error', 'evaluation_set', 'model_version']], 'price_predictions')
publish(report, 'model_metrics')
display(report)
print('Frozen model run ID:', run_id)
print('Label coverage:', data.groupby('city').price.agg(['count', 'size']))
display(predictions[predictions.evaluation_set.isin(['ab_test', 'c_unseen'])]
        .groupby(['city', 'room_type']).agg(n=('absolute_error', 'count'),
        model_mae=('absolute_error', 'mean'), baseline_mae=('baseline_absolute_error', 'mean')).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Refresh the dashboard / Genie views and table comments (`model/gold_views.sql`).

# COMMAND ----------

with open(os.path.abspath('../model/gold_views.sql')) as f:
    statements = [s.strip() for s in f.read().split(';')]
for statement in statements:
    body = '\n'.join(l for l in statement.splitlines() if not l.strip().startswith('--')).strip()
    if body:
        spark.sql(body.replace('airbnb_lakehouse.', f'{catalog}.'))
display(spark.table(f'{gold}.v_model_metrics').orderBy('run_id', 'city', 'variant'))

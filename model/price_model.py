"""Training-only preprocessing and baseline; no Spark dependency.

v1 (france_v1, week 1): random forest on raw EUR prices, 13 features. Kept so the frozen
artifact can still be scored.
v2 (week 2, after the Bordeaux run): random forest on log(price / city price level).
The city level is the median advertised price of the city's own training listings; for a
new city it comes from a calibration sample of that city's listings (see README, City C).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_VERSION = 'v2'

V1_NUMERIC = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenities_count',
              'minimum_nights', 'host_is_superhost', 'host_listings_count',
              'host_years', 'instant_bookable']
V1_CATEGORICAL = ['room_type', 'property_type', 'bathroom_type']

# instant_bookable dropped: empty in every 2026 snapshot. property_type dropped: its
# vocabulary means different things per city ("Entire home" is a rare luxury house in
# Paris and 18% of Bordeaux). Location and review features added; all are city-relative
# or city-neutral (distance is to the city's own median point).
NUMERIC = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenities_count',
           'minimum_nights', 'host_is_superhost', 'host_listings_count', 'host_years',
           'dist_to_center_km', 'review_scores_location', 'review_scores_rating',
           'number_of_reviews', 'availability_365']
CATEGORICAL = ['room_type', 'bathroom_type']
FEATURES = NUMERIC + CATEGORICAL
# Columns the notebook must load for either version.
INPUT_COLUMNS = sorted(set(FEATURES + V1_NUMERIC + V1_CATEGORICAL))
POOLED = '__pooled__'


def features(df, numeric=NUMERIC, categorical=CATEGORICAL):
    out = df[numeric + categorical].copy()
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
    for c in categorical:
        out[c] = out[c].fillna('__missing__').astype(str)
    return out


def forest(numeric, categorical):
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    return Pipeline([('prepare', preprocess), ('regressor', RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=5,
        criterion='squared_error', random_state=42, n_jobs=-1))])


def city_levels(df):
    """Median advertised price per city, plus the pooled median as the fallback for unknown cities."""
    priced = df[np.isfinite(df.price) & (df.price > 0)]
    levels = priced.groupby('city').price.median().astype(float).to_dict()
    levels[POOLED] = float(priced.price.median())
    return levels


def fit(train):
    """Fixed configuration chosen on an inner validation split of A/B training rows only."""
    levels = city_levels(train)
    pipeline = forest(NUMERIC, CATEGORICAL)
    pipeline.fit(features(train), np.log(train.price / train.city.map(levels)))
    model = {'version': MODEL_VERSION, 'pipeline': pipeline, 'levels': levels,
             'numeric': NUMERIC, 'categorical': CATEGORICAL}
    keys = train.assign(room_type=train.room_type.fillna('__missing__'),
                        accommodates=train.accommodates.fillna(-1))
    baseline = {
        'pairs': keys.groupby(['room_type', 'accommodates']).price.median().to_dict(),
        'rooms': keys.groupby('room_type').price.median().to_dict(),
        'global': float(train.price.median()),
    }
    return model, baseline


def predict(model, df, levels=None):
    """levels: city -> price level for cities not seen in training (e.g. a calibration sample)."""
    if isinstance(model, Pipeline):  # v1 artifact: raw-price forest
        return model.predict(features(df, V1_NUMERIC, V1_CATEGORICAL))
    known = {**model['levels'], **(levels or {})}
    level = df.city.map(known).fillna(known[POOLED]).to_numpy()
    relative = model['pipeline'].predict(features(df, model['numeric'], model['categorical']))
    return np.exp(relative) * level


def baseline_predict(df, baseline):
    rooms = df.room_type.fillna('__missing__')
    guests = df.accommodates.fillna(-1)
    return np.array([baseline['pairs'].get((r, a), baseline['rooms'].get(r, baseline['global']))
                     for r, a in zip(rooms, guests)])


def score(df, model, baseline, levels=None):
    out = df.copy()
    out['predicted_price'] = predict(model, df, levels)
    out['baseline_price'] = baseline_predict(df, baseline)
    out['residual'] = out.price - out.predicted_price
    out['absolute_error'] = abs(out.residual)
    out['baseline_absolute_error'] = abs(out.price - out.baseline_price)
    return out


def metrics(scored):
    rows = []
    groups = [(c, g) for c, g in scored.groupby('city')] + [('combined', scored)]
    for city, g in groups:
        g = g.loc[np.isfinite(g.price) & (g.price > 0)]
        if g.empty:
            raise ValueError('No valid evaluation labels for ' + city)
        model_mae, baseline_mae = g.absolute_error.mean(), g.baseline_absolute_error.mean()
        rows.append(dict(city=city, n=len(g), model_mae=float(model_mae),
                         baseline_mae=float(baseline_mae),
                         improvement_eur=float(baseline_mae-model_mae),
                         median_ae=float(g.absolute_error.median()),
                         mean_residual=float(g.residual.mean())))
    return pd.DataFrame(rows)

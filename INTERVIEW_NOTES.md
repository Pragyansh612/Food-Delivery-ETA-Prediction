# Interview prep notes

Straight answers to the questions this project is likely to draw, written
so I can say them out loud without looking anything up.

## Dataset overview

Kaggle "Food Delivery Dataset" (gauravmalik26), 45,593 raw rows of Indian
food delivery orders, Feb-Apr 2022. Each row: one order, with restaurant/
delivery coordinates, order timestamp, weather, traffic, delivery partner
age/rating, vehicle info, festival flag, city type, and the target
(delivery time in minutes). Pulled via a GitHub mirror of the same file
since I didn't have Kaggle API credentials set up in the dev environment
-- verified it's the identical dataset by checking the mirror repo's own
README, which links back to the same Kaggle page.

## EDA findings that mattered

- Target: mean 26.3 min, right-skewed, no negative/absurd values --
  clean target, no trimming needed.
- ~9% of rows had invalid coordinates (outside India, or at (0,0)) --
  this is the finding that shaped the biggest chunk of the cleaning step.
- Missingness was low (<4.2% per column) and spread across columns, not
  concentrated -- supported imputation over dropping.
- Traffic density showed a clean, near-monotonic relationship with
  delivery time before any modeling was done -- gave me confidence
  traffic was worth encoding carefully (ordinally, not one-hot).

## Features created and why

- `distance_km` (haversine): the most direct physical driver available.
- `order_hour` / `order_day_of_week` / `is_weekend`: time-of-day demand
  patterns (lunch/dinner rush) are a real, non-obvious driver of
  delivery time independent of raw traffic labels.
- `traffic_ordinal`: encoded as an ordered 0-3 scale, not one-hot,
  because Low<Medium<High<Jam is a real ordering and a linear model
  should be allowed to use it directly.
- `distance_x_traffic`: the one interaction term I added, with a
  specific hypothesis -- traffic costs more total minutes on a longer
  route, a multiplicative effect linear regression can't represent on
  its own. I tested this (refit without the term) and the effect was
  real but small (MAE 4.782 -> 4.779). If asked "why only one
  interaction feature," the honest answer is: I wanted each engineered
  feature to be tied to a concrete, checkable hypothesis rather than
  throwing combinations at the wall, and this was the one with the
  clearest mechanistic story.

## Leakage risks considered

The framing that drives every leakage decision: this model predicts ETA
**at order placement**, not mid-delivery.

- `Time_Order_picked`: doesn't exist yet at the moment we're predicting
  from -- it's generated once a courier reaches the restaurant. Including
  it would leak information from later in the process.
- `Delivery_person_ID`: risk of the model memorizing per-courier averages
  instead of learning a generalizable relationship, and it wouldn't work
  for a new courier. Noted a legitimate alternative (time-windowed
  historical courier average) as future work, not attempted here.
- Raw ID, raw date/time strings, raw lat/long: not leakage exactly, but
  redundant with derived features and risk overfitting to specific
  values seen in training rather than the general relationship.
- Added an `assert` in the notebook so any of these silently ending up
  back in the feature list would fail loudly.

## Why Linear Regression as the baseline

Because the point of a baseline isn't to be competitive -- it's to be a
floor. If Random Forest/XGBoost can't clear it by a real margin, the
extra complexity isn't earning its keep. It also gives an early, cheap
sanity check: if the "baseline" had come back with R2=0.95, that would
have been the moment to stop and check for leakage, before ever training
a tree model.

## Why Random Forest/XGBoost performed better (or didn't)

Both beat linear regression by a wide margin (R2 0.605 -> ~0.84) because
they can model non-linear relationships and interactions (like
distance x traffic) without needing them hand-engineered. Random Forest
edged out XGBoost very slightly (MAE 3.097 vs 3.126) -- I don't read that
as "Random Forest is better than XGBoost" in general; with only light
tuning (small grid, cv=3, per the project's own time-boxing) on ~32k
training rows, a 0.03 MAE gap is within the noise I'd expect between two
models of comparable capacity. I picked Random Forest as the final model
because of that marginal edge plus lower sensitivity to hyperparameter
choices, not because it's categorically the stronger algorithm.

## Biggest limitations of the dataset

- Self-reported/categorical traffic and weather, not live sensor data.
- Straight-line (haversine) distance, not real road distance or GPS.
- Distance values look banded/discretized -- likely partially synthetic
  data, which probably makes the achieved R2 higher than a live system
  with noisier signals would produce. I'd treat the model *ranking*
  (tree models beat linear by a lot) as the durable takeaway, and treat
  the absolute R2=0.84 as specific to this dataset's cleanliness -- a
  caveat I'd raise unprompted, not wait to be asked about.
- Single region (India) and 8-week window -- no seasonality beyond one
  festival flag, no other-country delivery patterns.
- No restaurant-side signal (prep time, kitchen load) which plausibly
  matters in reality.

## Three improvements with more time/data

1. **Time-windowed courier-performance features** (e.g. a courier's
   trailing 30-day average delivery time, computed only from orders
   before the current one) instead of excluding `Delivery_person_ID`
   entirely -- captures real courier-level signal without the
   memorization risk of a raw ID.
2. **Real road distance/ETA from a routing API** instead of haversine,
   which would remove the "as the crow flies" approximation and likely
   improve the model meaningfully, especially in areas with rivers,
   highways, or one-way systems that haversine can't see.
3. **A longer, multi-region data collection window** to check whether
   the traffic/weather/distance relationships found here generalize, or
   are specific to this city mix and this 8-week period -- the current
   single 8-week India-only window makes it impossible to tell those
   apart.

# Progress log

Working log for the food delivery ETA prediction case study. Bullet points,
timestamps, honest notes -- not a polished doc.

## 2026-07-08

- **Setup.** Repo init, venv, requirements.txt (pandas, numpy, scikit-learn,
  xgboost, matplotlib, seaborn, jupyter/nbformat/nbclient), folder structure.
  Committed as `init project structure`.

- **Data acquisition.** Needed a public food delivery ETA dataset, 20k+ rows.
  Target: Kaggle "Food Delivery Dataset" (gauravmalik26,
  https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset).
  No Kaggle API credentials configured in this environment (no
  `~/.kaggle/kaggle.json`, no env vars, `kaggle` CLI not installed), so
  couldn't pull it directly via the Kaggle API. Found the identical file
  mirrored in a public GitHub repo
  (github.com/Vikranth3140/Food-Delivery-Time-Prediction,
  `datasets/kaggle/train.csv`) whose own README links back to the same
  Kaggle dataset page, confirming it's the same source data, not a
  substitute. Downloaded via raw.githubusercontent.com and saved to
  `data/raw/food_delivery_train.csv`. 45,593 rows, 20 columns.
  Committed as `add raw dataset and loader script`.

- **EDA.** Notebook section 2. Key findings:
  - Target (`Time_taken(min)`) is stored as a string with a literal
    `'(min) '` prefix. Parsed: mean 26.3 min, median 26, std 9.4,
    range 10-54, right-skewed (skew 0.49). No negative/absurd values.
  - Most "numeric-looking" columns (age, ratings, multiple_deliveries) are
    actually string dtype because missing values are encoded as the
    literal string `'NaN '` (with trailing space) rather than a real NaN,
    and every categorical has trailing whitespace baked in
    (`'Urban '`, `'High '`). This is raw, unprocessed data -- confirms
    we're not working with a pre-cleaned toy dataset.
  - Missingness: 7 columns affected (ratings, age, order time, city,
    multiple_deliveries, traffic, festival), all between 0.5% and 4.2%.
    Spread across columns rather than concentrated -- looks like random
    per-field reporting gaps, not something correlated with the target.
    Going with simple imputation in cleaning rather than dropping rows.
  - Coordinates: 4,071 rows have restaurant coordinates outside
    mainland India's lat/long bounds or sitting at/near (0,0) (a known
    bad-geocode placeholder pattern). These get dropped in cleaning --
    a wrong restaurant location corrupts the single most important
    engineered feature (distance).
  - Ratings: 53 rows have a rating > 5 (max seen: 6.0), which is out of
    the valid 1-5 range for a star rating. Flagged for cleaning.
  - Distance (quick haversine calc, not yet the final feature) has a
    real but moderate correlation with delivery time (r=0.32 on valid
    rows). Traffic density shows a clean monotonic step-up from Low to
    Jam. Weather shows smaller but sensible separation (Sunny fastest,
    Fog/Cloudy slowest).
  - Noticed: distance values look banded/discretized (repeating at
    round-ish km values) rather than continuous GPS-derived distances.
    Suggests the underlying coordinates may be partially synthetic/
    simulated rather than raw GPS traces. Flagging as a dataset
    limitation for the README rather than treating it as a bug to fix.
  - Saved 4 plots to `reports/figures/`: target distribution,
    missingness by column, outlier boxplots (target + age), and a
    distance/traffic-vs-time panel.
  - Committed as `EDA: target distribution and missingness`.

- **Cleaning.** Notebook section 3, mirrored in `src/clean.py`. Order of
  operations: parse strings/target first, then drop unrecoverable rows,
  then impute what's left.
  - Dropped 4,071 rows with invalid restaurant/delivery coordinates
    (outside India bounds or (0,0) placeholder).
  - Of what remained, dropped 24 more rows with out-of-range ratings
    (>5) -- most of the original 53 bad-rating rows had already been
    removed by the coordinate filter (overlap).
  - Dropped 1,301 rows with missing `Time_Orderd`. Decided against
    imputing a timestamp -- there's no defensible "typical" order time
    to fabricate, and this field feeds directly into the hour-of-day
    feature.
  - Imputed the remaining low-missingness fields: age/ratings/
    multiple_deliveries with median, traffic/festival/city with mode.
    All under ~4% missing among what was left, spread randomly, so
    simple imputation is defensible.
  - Net: 45,593 -> 40,197 rows (88.2% kept). Zero missing values
    remaining. Verified `src/clean.py` reproduces the exact same numbers
    as the notebook when run standalone.
  - Note: there were no negative durations in this dataset (target min
    was 10 minutes) -- the actual bad-value issue here was invalid
    coordinates and out-of-range ratings, not negative durations.
  - Committed as `drop invalid coordinates and bad ratings`.

- **Feature engineering.** Notebook section 4, mirrored in `src/features.py`.
  - `distance_km`: haversine between restaurant and delivery coordinates.
    Range 1.5-21km after cleaning, mean 9.7km. Values still look banded
    (same observation as EDA) but that's a data characteristic, not a
    bug in the calculation.
  - `order_hour`, `order_day_of_week`, `is_weekend` from `Order_Date` +
    `Time_Orderd`. Hour distribution shows a clear lunch (11-12) and
    dinner (17-23) concentration -- matches real delivery demand
    patterns, which is a good sanity check that the timestamp parsing
    is correct.
  - `traffic_ordinal`: ordinal-encoded traffic (Low<Medium<High<Jam)
    instead of one-hot, since it's a genuinely ordered variable and a
    linear model should be allowed to use that order directly.
  - One-hot encoded weather, order type, vehicle type, city (no natural
    order in any of these).
  - `festival_flag`: binary Yes/No.
  - One deliberate interaction feature: `distance_x_traffic` (distance
    times traffic ordinal). Reasoning: a traffic jam costs more total
    minutes on a longer route than a short one -- a multiplicative
    effect a plain linear model can't represent from the two features
    separately, but that tree models can already discover on their own
    via splits. This gives a concrete, checkable hypothesis for later:
    the interaction term should help Linear Regression more than it
    helps Random Forest/XGBoost. Deliberately did not add more
    interaction terms beyond this one to avoid feature bloat "for show".
  - Final feature set: 40 columns (from 20 raw). Verified
    `src/features.py` reproduces the same shape/columns as the notebook.

- **Leakage review.** Notebook section 5. Framed the prediction problem
  explicitly: ETA at the moment the order is placed, not a mid-delivery
  update. That framing is what decides what counts as leakage.
  - Excluded as genuine leakage risk: `Time_Order_picked` (doesn't exist
    yet at order-placement time -- it's generated once a courier reaches
    the restaurant, downstream of the prediction point) and
    `Delivery_person_ID` (risk of the model memorizing per-courier
    historical averages instead of learning generalizable relationships;
    a properly time-windowed courier-average feature would be a
    legitimate future addition but wasn't attempted here).
  - Excluded for redundancy/generalization (not leakage): `ID`, raw
    `Order_Date`/`Time_Orderd` strings (superseded by derived hour/day/
    weekend features), raw lat/long columns (superseded by
    `distance_km`, to avoid overfitting to specific coordinate clusters
    seen in training).
  - Added an explicit `assert` in the notebook that none of the
    excluded columns end up in the final feature list -- cheap
    insurance against accidentally leaving one in.
  - Final feature count: 27.

- **Train/test split.** Notebook section 6, mirrored in `src/split.py`.
  Data spans 2022-02-11 to 2022-04-06 (~8 weeks) so did a time-based
  split at the 80th percentile date (2022-03-29) rather than a random
  shuffle: train = everything before that date (31,672 rows), test =
  everything on/after (8,525 rows, 21.2%). Chose this over random
  shuffling because it mirrors actual deployment (train on past, predict
  future) and avoids a subtler leakage: a random shuffle could let
  same-week/same-pattern orders sit in both train and test, making the
  model look better than it would on genuinely unseen future orders.
  Train/test target means are close (26.25 vs 26.59), so no obvious
  regime shift between the two periods.

- **Baseline: Linear Regression.** Notebook section 7, mirrored in
  `src/train_baseline.py` + `src/evaluate.py`. Standardized features,
  fit on train, evaluated on the held-out time-based test set.
  **MAE=4.779, RMSE=5.981, R2=0.605.** Predicted-vs-actual scatter looks
  like a real linear fit (band around the diagonal, more scatter at the
  extremes) -- nothing suspicious. R2 of 0.6 is in the range I'd expect
  for a model using only order-time metadata (no live GPS, no real-time
  prep-queue signal), not implausibly high. This is the number every
  other model gets compared against.

- **Random Forest + XGBoost.** Notebook section 8, mirrored in
  `src/train_models.py`. Light GridSearchCV (small grids, cv=3,
  scoring=neg MAE), unscaled features (trees don't need scaling).
  - Random Forest best params: max_depth=12, n_estimators=200.
    **MAE=3.097, RMSE=3.802, R2=0.840** on test.
  - XGBoost best params: learning_rate=0.05, max_depth=6,
    n_estimators=400. **MAE=3.126, RMSE=3.863, R2=0.835** on test.
  - Both tree models land close together and both clearly beat the
    linear baseline (MAE 4.78 -> ~3.1, R2 0.605 -> ~0.84). RF edges out
    XGBoost very slightly here -- within the range I'd expect from two
    models with comparable capacity and only light tuning on tabular
    data this size; not going to over-read a 0.03 MAE gap as "RF is
    better than XGBoost" in general.
  - Went back and tested the `distance_x_traffic` hypothesis from
    feature engineering (that it should help linear regression more
    than trees): refit linear regression without the term. MAE went
    from 4.779 to 4.782 -- a real but genuinely tiny effect, not the
    dramatic difference I'd have liked to report. Being honest about
    this rather than dropping the check because it wasn't impressive:
    the interaction reasoning was sound, the practical impact is just
    small in this dataset.

- **Final evaluation.** Notebook section 9.
  - Comparison table: Linear Regression (MAE 4.779, RMSE 5.981, R2
    0.605), Random Forest (MAE 3.097, RMSE 3.802, R2 0.840), XGBoost
    (MAE 3.126, RMSE 3.863, R2 0.835). Picked **Random Forest** as final
    model -- marginally better test MAE, and comparably strong with
    only light tuning.
  - 5-fold CV on the training set (test set untouched): MAE
    3.029 +/- 0.019, consistent with test MAE of 3.097 -- no sign the
    single train/test split got lucky or unlucky.
  - Feature importance (RF): top features are `distance_x_traffic`
    (0.221), `Delivery_person_Ratings` (0.204), `Delivery_person_Age`
    (0.106), `Vehicle_condition` (0.106), `weather_Sunny` (0.088),
    `distance_km` (0.087). Surprise worth noting honestly: raw
    `traffic_ordinal` importance is low (0.023) even though EDA showed
    a clean traffic-vs-time relationship -- explained by
    `distance_x_traffic` already carrying that signal, so the tree
    splits on the interaction term instead of the raw traffic column
    (importance reflects "which feature got split on", not standalone
    causal weight). Courier rating and vehicle condition mattering more
    than raw distance was not what I expected going in, but it's a
    plausible operational story (rating/vehicle condition as proxies
    for real-world speed and reliability), not a red flag.
  - **Leakage sanity check** (R2=0.84 is a big jump over the 0.605
    baseline, explicitly investigated per the project rules rather than
    accepted at face value): train MAE (2.532) vs test MAE (3.097) shows
    normal RF overfit behavior, not a cliff; no single feature or top-2
    combo dominates importance (top 2 sum to 0.53, not near-1.0, which
    is what you'd expect if a target-proxy had snuck in); the excluded-
    column assert from step 5 re-confirmed. **Conclusion: no leakage
    found** -- the RF/XGBoost improvement over linear regression comes
    from capturing non-linear and interaction effects, not from a
    mishandled column. Separately flagging (not a leakage issue, an
    honesty-about-the-number issue): distance values are banded/
    discretized rather than continuous, suggesting this dataset may be
    partially synthetic/simulated rather than raw GPS traces -- which
    means the 0.84 R2 here likely reflects unusually clean underlying
    relationships in this specific dataset, and shouldn't be assumed to
    carry over to a live system with noisier real-world GPS/traffic
    data. Noting this explicitly in the README limitations.

**Core ETA project (Product.md steps 1-10) is complete.** Next: README
and INTERVIEW_NOTES.md (step 12), then -- only if time remains -- the
optional recommendation stretch goal (step 11).

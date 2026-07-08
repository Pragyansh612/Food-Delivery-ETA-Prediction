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

**Next up:** feature engineering (step 5) -- haversine distance, time-of-day
features, categorical encoding.

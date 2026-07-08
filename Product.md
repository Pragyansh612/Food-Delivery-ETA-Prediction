### 3.2 Work to be done, in order

1. **Setup** — repo init, venv, requirements.txt, .gitignore, folders.
2. **Data acquisition** — download dataset, save to `data/raw/`, document
   source in README as you go.
3. **EDA** — target distribution, missingness, outliers, correlation
   between distance/traffic/weather and delivery time. Save 3-5 plots.
4. **Cleaning** — handle missing values, fix inconsistent categories,
   remove invalid coordinates/durations. Note *why* each row-drop or
   imputation decision was made — this becomes interview material.
5. **Feature engineering**:
   - Haversine distance between restaurant and delivery location
   - Hour of day, day of week, is_weekend from order timestamp
   - Encode categorical variables (traffic, weather, vehicle type)
   - Only add interaction features that are genuinely justified, not for
     show — be ready to explain why each feature is there
6. **Leakage check** — explicitly list, in a code comment or notebook
   markdown cell, which columns were excluded and why (e.g. anything
   only known after delivery). This is a common interview question —
   answer it in the artifact itself.
7. **Train/test split** — proper split (time-based if timestamps allow).
8. **Baseline model** — Linear Regression, report MAE/RMSE/R².
9. **Model comparison** — Random Forest and XGBoost, light tuning
   (GridSearchCV or manual — this is a case study, not a Kaggle
   leaderboard push).
10. **Evaluation** — comparison table across models. Cross-validate the
    best model. Plot feature importance. Sanity-check: if any result
    looks unrealistically good, stop and investigate leakage before
    moving on — do not silently accept "too good" numbers.
11. **Optional stretch (only if time remains after step 10 is fully
    done and committed):** a small restaurant recommendation module
    using collaborative filtering on implicit feedback (e.g. order
    history as implicit rating). Keep it genuinely small — a working
    notebook section, not a second full repo. Do not start this until
    the ETA prediction project is complete, documented, and committed.
12. **README** — see section 4.
13. **Commit after every step**, with specific, human-sounding commit
    messages. Example sequence:
    - `init project structure`
    - `add raw dataset and loader script`
    - `EDA: target distribution and missingness`
    - `drop invalid coordinates and negative durations`
    - `add haversine distance and time-based features`
    - `document excluded columns to avoid leakage`
    - `baseline linear regression`
    - `random forest + xgboost comparison`
    - `feature importance and final evaluation`
    - `write README with results`
14. **Maintain PROGRESS.md** — after each step, append a short log entry:
    what was done, what was found, what's next. This is a working log,
    not a polished doc — bullet points, timestamps, honest notes
    ("tried X, didn't help, reverted" is a good entry).

## 4. README requirements

Written like the person who actually did the work — plain, specific,
numbers-driven, slightly informal. No "leveraging cutting-edge techniques,"
no emoji, no wall of vague bullets.

Must include:
- The business framing from section 1, in your own words, 2-3 sentences
- Dataset source, size, and what each row represents
- 3-4 specific EDA findings with real numbers
- Feature engineering summary and the leakage-avoidance decisions
- Model comparison table (Linear Regression vs Random Forest vs XGBoost —
  MAE, RMSE, R²)
- Best model and a real reason why (not just "it had the best number")
- Feature importance plot with interpretation
- Honest limitations (e.g. self-reported traffic, no live GPS, dataset
  from a specific region/time period)
- If a recommendation module was added: 3-4 sentences on how it works
  and what it would need to become production-ready
- How to run it locally

## 5. Code style requirements

- Comments only where a decision needs explaining, not on obvious lines.
- Slightly human variable naming — readable but not textbook-uniform but make sure its optimized for clean, readable and         interviewable.
- No emoji anywhere, in code or commits.
- Docstrings only on non-trivial functions.
- Functions reasonably sized — no single 300-line script.
- A few interactive-style print/sanity-check statements in the notebook,
  the way someone actually working through it would leave in.

## 6. Non-negotiable rules

- **No fabricated or cosmetically tuned metrics.** Report whatever the
  model actually achieves. If MAE is mediocre, say so and explain why in
  the limitations section — that is a more credible interview answer than
  a suspiciously clean number.
- **No placeholder numbers anywhere** in README, notebook, or code
  comments. Every number must come from code that was actually executed.
- **Prioritize defensibility over the last 1-2% of metric improvement.**
  Every choice (split strategy, feature, model, hyperparameter) must be
  something the author can explain out loud in an interview.
- If something in this spec is ambiguous, make the decision a working
  data scientist would make, and record the reasoning in the relevant
  commit message or PROGRESS.md entry — don't stop to ask.

## 7. Explicit non-goals

- No deployment, no API, no frontend for this version.
- No deep learning — tree-based models are the right tool here.
- No Docker, no CI/CD, no test suite — this is a focused case study.
- No second dataset merged in "for more data" — one clean dataset, well
  understood, beats two stitched together.

## 8. Success criteria (what "done" looks like)

- Repo runs end-to-end from `requirements.txt` + README instructions.
- Every metric in the README is reproducible from the notebook/scripts.
- README and commit history read like real incremental work, not one dump.
- The author can answer, without looking anything up:
  - Why this train/test split
  - Why XGBoost did or didn't beat Random Forest / Linear Regression
  - What leakage risks were considered and how they were avoided
  - Which feature mattered most and a plausible operational reason why
  - What they'd do differently with more time/data

## Interview Preparation Notes

Create INTERVIEW_NOTES.md containing:

- Dataset overview
- Important EDA findings
- Features created and why
- Leakage risks considered
- Why Linear Regression was used as baseline
- Why XGBoost/Random Forest performed better or worse
- Biggest limitations of the dataset
- Three improvements that would be made with more time/data

Do not spend more than 20-30 minutes tuning hyperparameters.
A completed project with solid explanations is preferred over marginal metric improvements.
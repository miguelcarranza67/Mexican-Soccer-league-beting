# Mexican-Soccer-league-beting
Models used with claude's help to predict soccer games
Liga MX Match Predictor: Poisson Regression & Gradient Boosting

A set of three progressively-refined statistical models that predict match outcomes (final score probabilities) for Liga MX, Mexico's top professional soccer division. The project started as a simple recreation of a Poisson-based prediction model I saw online, and evolved into a more robust pipeline as I diagnosed its weaknesses and fixed them one at a time — real historical data, live match updates, time-decay weighting, a known statistical correction for low-scoring games, and a recent-form feature for the machine learning model.

Each match is modeled as two goal counts (home team, away team), each following a Poisson distribution whose expected value (λ) depends on each team's attacking/defensive strength. From there, a full score matrix (0-0, 1-0, 2-1, etc.) is built, from which win/draw/loss probabilities are derived.

Model 1 — Baseline (liga_mx_poisson_xgboost.py)

Predicts: Win/draw/loss probability and most likely final scores for any two Liga MX teams.

Approach:

Poisson regression — a statistical model (fit with statsmodels) that learns one "attack strength" and one "defense strength" number per team, plus a single "home advantage" number, by finding the values that best explain historical goal counts. It's the classic academic approach to football modeling (Maher, 1982).
XGBoost — a machine learning model (gradient-boosted decision trees) trained with a Poisson objective, so instead of assuming a fixed formula, it learns patterns in the data directly. Its output (expected goals) feeds into the same Poisson score-matrix logic as Model 1.

Data: ~2,047 real Liga MX matches (2018–19 through 2024–25 seasons), pulled from an open-source GitHub repository of historical football results (team names, dates, final scores).

Results: Both models produce sensible, comparable probabilities for real fixtures (e.g., ~42–48% win probability for the stronger side in mid-table matchups). No formal backtesting was run against this version — its main value was proving the pipeline worked end-to-end before improving it.

What I learned building it: with too little data (an early prototype trained on only ~10 matches) the model becomes unstable and produces nonsensical probabilities — a hands-on lesson in why sample size matters for maximum-likelihood estimation.

Model 2 — Updated Baseline (liga_mx_original_actualizado.py)

Predicts: Same as Model 1, same two techniques (Poisson regression + XGBoost).

Approach: Identical modeling logic to Model 1 — deliberately kept simple, with no time-weighting, so every match (from 2018 or from last week) counts equally. It exists as a controlled comparison point: isolating the effect of "more/fresher data" from the effect of the improvements added in Model 3.

Data: Same historical GitHub dataset, extended with matches from 2024-25 through the present, pulled live from ESPN's public scoreboard API, with team-name reconciliation between the two sources (e.g., "Atlas" → "Atlas Guadalajara") and a local cache to avoid re-fetching on every run.

Results: Produces noticeably different probabilities than Model 1 on the same fixtures once recent results are included, confirming that adding current-season data changes the model's read on team strength — the first quantitative signal that recency matters.

What changed from Model 1: only the data window (more/fresher matches). No changes to the modeling technique itself — this was intentional, to test one variable at a time.

Model 3 — Full Pipeline (liga_mx_poisson_xgboost_v3.py)

Predicts: Same as above, with more refined probabilities.

Approach: Same two base models, plus three statistical improvements:

Recency weighting — older matches count less using exponential time-decay, since a team's strength today isn't well represented by a 2018 result.
Dixon-Coles correction — a published statistical fix (Dixon & Coles, 1997) for a known bias where basic Poisson models slightly mis-predict low-scoring outcomes (0-0, 1-0, 0-1, 1-1); an extra parameter is estimated to correct just those four cells.
Recent-form feature (XGBoost only) — a rolling average of each team's points/goals over its last 5 matches, computed carefully to only use past matches relative to the one being predicted (avoiding data leakage).

Data: Same combined GitHub + ESPN dataset as Model 2 (~2,700+ matches and growing), with the live-fetch step cached locally for faster reruns.

Results: On several test fixtures, Model 3's two internal models (Poisson vs. XGBoost) sometimes disagree with each other on a close match (e.g., one favoring León, the other favoring Pachuca in the same fixture) — which I found to be genuinely useful information: when both approaches agree, confidence is higher; when they split, it flags a genuinely uncertain matchup. I also used the model's output probabilities to walk through expected-value and Kelly-criterion calculations for betting scenarios, as a practical application of the probabilities it produces.

What changed from Model 2: three targeted statistical fixes, each one isolated and verified independently before moving to the next, rather than changing everything at once.

Known limitations / next steps: no formal backtesting (accuracy, Brier score, calibration curves) has been run yet against held-out real results — this is the natural next step to quantify how well-calibrated the probabilities actually are. A Random Forest variant (bagging, as opposed to XGBoost's boosting) is also planned as a further comparison point.

How I used Claude

I built this project through an extended, hands-on session with Claude, using it as an active collaborator rather than a code generator:

Debugging, step by step: every error I hit in Google Colab (indentation bugs, wrong data formats, mismatched team names between data sources, XGBoost sample-weight mismatches) was diagnosed and fixed interactively, with Claude explaining why each error happened, not just the fix.
Data sourcing under real constraints: when the original GitHub dataset turned out to only cover Mexico from 2018–19 onward, Claude helped me investigate the gap, evaluate alternative sources (Wikipedia, fbref, ESPN's API), and settle on a working solution.
Statistical design: Claude explained and helped implement the Dixon-Coles correction, the recency-weighting scheme, and the leakage-safe recent-form feature — translating published statistical methods into working code and explaining the reasoning behind each one.
Critical evaluation, not just code: when I asked about using the model's output for betting decisions, Claude walked through the actual expected-value math and the Kelly criterion rather than just answering yes/no, and flagged the model's real limitations (small effective sample size, no live roster/manager data) instead of overselling its accuracy.
Tech Used
Python 3 (Google Colab)
pandas, numpy — data cleaning and manipulation
statsmodels — Poisson regression (GLM)
scipy — probability distributions (Poisson) and numerical optimization (Dixon-Coles ρ estimation)
xgboost — gradient-boosted trees with a Poisson objective
matplotlib — score-matrix heatmap visualizations
requests — live data fetching from ESPN's public API
Data sources: footballcsv/mexico (historical results), ESPN public scoreboard API (recent/live results)

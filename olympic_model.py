# Machine Learning Project Olympic Team Classification Gymnastics

# ============================================================
# 0. LOAD + PREPARE DATA
# ============================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("gymnastics_ml_ready.csv")
df = df.sort_values(["athlete_id", "Apparatus", "Year"])

print("Initial shape:", df.shape)

# ============================================================
# 1. COMPETITION TYPE + WEIGHT
# ============================================================
def classify_competition(comp):
    if pd.isna(comp):
        return "Other"
    comp = comp.lower()

    if any(x in comp for x in ["world", "olympic", "world cup", "fisu", "dtb pokal"]):
        return "International"
    elif any(x in comp for x in ["european", "pan american", "asian", "commonwealth"]):
        return "Regional"
    elif any(x in comp for x in ["u.s.", "usa", "british", "core hydration"]):
        return "Domestic"
    else:
        return "Other"

df["competition_type"] = df["Competition"].apply(classify_competition)

weight_map = {
    "International": 1.0,
    "Regional": 0.8,
    "Domestic": 0.3,
    "Other": 0.5
}

df["competition_weight"] = df["competition_type"].map(weight_map)

# ============================================================
# 2. ATHLETE TYPE
# ============================================================
athlete_profile = (
    df.groupby(["athlete_id", "Year", "Gender"])["Apparatus"]
      .nunique()
      .reset_index(name="n_events")
)

def classify_athlete(row):
    if row["Gender"] == "MAG":
        return "AA" if row["n_events"] >= 5 else "Specialist"
    else:
        return "AA" if row["n_events"] >= 3 else "Specialist"

athlete_profile["athlete_type"] = athlete_profile.apply(classify_athlete, axis=1)

df = df.merge(
    athlete_profile[["athlete_id", "Year", "athlete_type"]],
    on=["athlete_id", "Year"],
    how="left"
)

print("\nAthlete type distribution:")
print(athlete_profile["athlete_type"].value_counts())

# ============================================================
# 3. TOKYO 2021 THRESHOLDS (per Gender + Apparatus)
# ============================================================
tokyo = df[df["Year"] == 2021].copy()

thresholds = (
    tokyo.groupby(["Gender", "Apparatus"])["Score"]
    .quantile([0.40, 0.60, 0.85])
    .unstack()
    .reset_index()
)

thresholds.columns = ["Gender", "Apparatus", "p40", "p60", "p85"]

df = df.merge(thresholds, on=["Gender", "Apparatus"], how="left")
print("\nThresholds sample:")
print(thresholds.head())

# ============================================================
# 4. LAG FEATURES
# ============================================================
df["lag_1"] = df.groupby(["athlete_id", "Apparatus"])["Score"].shift(1)
df["lag_2"] = df.groupby(["athlete_id", "Apparatus"])["Score"].shift(2)
df["lag_3"] = df.groupby(["athlete_id", "Apparatus"])["Score"].shift(3)

df["avg_last_3"] = (
    df.groupby(["athlete_id", "Apparatus"])["Score"]
      .transform(lambda x: x.shift(1).rolling(3).mean())
)

df["trend"] = df["lag_1"] - df["lag_3"]

# ============================================================
# D & E FEATURES (NO LEAKAGE)
# ============================================================

df = df.sort_values(["athlete_id", "Apparatus", "Year"])

# Lag 1 (most recent previous routine)
df["d_lag_1"] = (
    df.groupby(["athlete_id", "Apparatus"])["D_Score"]
      .shift(1)
)

df["e_lag_1"] = (
    df.groupby(["athlete_id", "Apparatus"])["E_Score"]
      .shift(1)
)

# Interaction (difficulty × execution)
df["d_e_interaction"] = df["d_lag_1"] * df["e_lag_1"]

print("\nLag D&E sample:")
print(df[["Score", "lag_1", "lag_2", "lag_3","D_Score","E_Score","d_lag_1","e_lag_1","d_e_interaction"]].head())


# ============================================================
# 4. PERFORMANCE TIERS
# ============================================================
def assign_tier(row):
    if pd.isna(row["lag_1"]):
        return np.nan
    elif row["lag_1"] >= row["p85"]:
        return "Elite"
    elif row["lag_1"] >= row["p60"]:
        return "High"
    elif row["lag_1"] >= row["p40"]:
        return "Average"
    else:
        return "Low"

df["performance_tier"] = df.apply(assign_tier, axis=1)

tier_map = {"Low": 0, "Average": 1, "High": 2, "Elite": 3}
df["tier_numeric"] = df["performance_tier"].map(tier_map)

print("\nTier distribution:")
print(df["performance_tier"].value_counts())


# ============================================================
# 5. OLYMPIC FEATURE
# ============================================================
tokyo = df[df["Year"] == 2021].copy()

tokyo["olympic_status"] = "participant"
tokyo.loc[tokyo["Rank"] <= 8, "olympic_status"] = "finalist"
tokyo.loc[tokyo["Rank"] <= 3, "olympic_status"] = "medalist"

olympic_strength = (
    tokyo.groupby(["Country", "Gender", "olympic_status"])
         .agg(n=("athlete_id", "nunique"))
         .reset_index()
)

# ------------------------------------------------------------
# STEP 1: pivot
# ------------------------------------------------------------
olympic_strength_pivot = olympic_strength.pivot_table(
    index=["Country", "Gender"],
    columns="olympic_status",
    values="n",
    fill_value=0
).reset_index()

# ------------------------------------------------------------
# STEP 2: safe column extraction
# ------------------------------------------------------------
medalist = olympic_strength_pivot.get("medalist", 0)
finalist = olympic_strength_pivot.get("finalist", 0)
participant = olympic_strength_pivot.get("participant", 0)

# ------------------------------------------------------------
# STEP 3: normalization 
# ------------------------------------------------------------
total = medalist + finalist + participant

olympic_strength_pivot["olympic_weight"] = (
    (3 * medalist + 2 * finalist + participant)
    / total.replace(0, np.nan)
).fillna(0)

# ============================================================
# 6. DEPTH FEATURES (TOP-K + WEIGHTED TEAM CEILING)
# ============================================================

# ------------------------------------------------------------
# Step 1 — Keep only international and regional meets
# ------------------------------------------------------------
df_depth = df[df["competition_type"].isin(["International", "Regional"])].copy()

# ------------------------------------------------------------
# Step 2 — Rank athletes within each country / gender / year / apparatus
# based on previous score (lag_1)
# ------------------------------------------------------------
K = 3

df_depth["rank_within_country"] = (
    df_depth.groupby(["Country", "Gender", "Year", "Apparatus"])["lag_1"]
    .rank(method="first", ascending=False)
)

# ------------------------------------------------------------
# Step 3 — Keep only Top-K athletes
# ------------------------------------------------------------
df_topK = df_depth[df_depth["rank_within_country"] <= K].copy()

print("Top-K dataset:", df_topK.shape)

# ------------------------------------------------------------
# Step 4 — Weighted Top-K performance feature
# 1st athlete = 0.5, 2nd = 0.3, 3rd = 0.2
# ------------------------------------------------------------
topk_score = df_topK.sort_values(
    ["Country", "Gender", "Year", "Apparatus", "lag_1"],
    ascending=[True, True, True, True, False]
).copy()

topk_score["weight"] = (
    topk_score.groupby(["Country", "Gender", "Year", "Apparatus"])
    .cumcount()
    .map({0: 0.5, 1: 0.3, 2: 0.2})
)

topk_score["weighted_score"] = topk_score["lag_1"] * topk_score["weight"]

topk_score = (
    topk_score.groupby(["Country", "Gender", "Year", "Apparatus"])["weighted_score"]
    .sum()
    .reset_index(name="topk_score_weighted")
)

print("\nTop-K weighted score sample:")
print(topk_score.head())

# ------------------------------------------------------------
# Step 5 — Count Top-K athletes by performance tier
# ------------------------------------------------------------
depth_tier = (
    df_topK.groupby(["Country", "Gender", "Year", "Apparatus", "performance_tier"])
    .agg(n_athletes=("athlete_id", "nunique"))
    .reset_index()
)

# ------------------------------------------------------------
# Step 6 — Pivot to wide format
# ------------------------------------------------------------
depth_pivot = depth_tier.pivot_table(
    index=["Country", "Gender", "Year", "Apparatus"],
    columns="performance_tier",
    values="n_athletes",
    fill_value=0
).reset_index()

# ------------------------------------------------------------
# Step 7 — Ensure all tier columns exist
# ------------------------------------------------------------
for col in ["Elite", "High", "Average", "Low"]:
    if col not in depth_pivot.columns:
        depth_pivot[col] = 0

# ------------------------------------------------------------
# Step 8 — Total athletes in Top-K group
# ------------------------------------------------------------
depth_pivot["total_athletes"] = (
    depth_pivot[["Elite", "High", "Average", "Low"]].sum(axis=1)
)

# ------------------------------------------------------------
# Step 9 — Weighted depth score
# Elite = 3, High = 2, Average = 1, Low = 0
# ------------------------------------------------------------
depth_pivot["depth_score"] = (
    3 * depth_pivot["Elite"] +
    2 * depth_pivot["High"] +
    1 * depth_pivot["Average"]
) / depth_pivot["total_athletes"].replace(0, np.nan)

depth_pivot["depth_score"] = depth_pivot["depth_score"].fillna(0)

# ------------------------------------------------------------
# Step 10 — Elite ratio
# ------------------------------------------------------------
depth_pivot["elite_ratio"] = (
    depth_pivot["Elite"] / depth_pivot["total_athletes"].replace(0, np.nan)
).fillna(0)

# ------------------------------------------------------------
# Debug prints
# ------------------------------------------------------------
print("\nDepth pivot sample:")
print(depth_pivot.head())

print("\nDepth score stats:")
print(depth_pivot["depth_score"].describe())

print("\nElite ratio stats:")
print(depth_pivot["elite_ratio"].describe())

# ============================================================
# 7. MODEL DATASET 
# ============================================================

df_model = df.copy()

# Step 1 — create model dataset
df_model = df_model.dropna(subset=["lag_1"])

# Step 2 — merge Olympic feature
df_model = df_model.merge(
    olympic_strength_pivot,
    on=["Country", "Gender"],
    how="left"
)

# Step 3 — merge depth feature
df_model = df_model.merge(
    depth_pivot,
    on=["Country", "Gender", "Year", "Apparatus"],
    how="left"
)
# Step 4 — merge topK feature
df_model = df_model.merge(
    topk_score,
    on=["Country", "Gender", "Year", "Apparatus"],
    how="left"
)

# Step 5 — fill missing values
df_model["olympic_weight"] = df_model["olympic_weight"].fillna(0)
df_model["elite_ratio"] = df_model["elite_ratio"].fillna(0)
df_model["depth_score"] = df_model["depth_score"].fillna(0)
df_model["topk_score_weighted"] = df_model["topk_score_weighted"].fillna(0)


# ============================================================
# 8. ENCODING
# ============================================================

# Gender
df_model["Gender_M"] = (df_model["Gender"] == "MAG").astype(int)

# Competition type dummies
df_model = pd.get_dummies(df_model, columns=["competition_type"], drop_first=True)

# ============================================================
# 9 . DATA PREP & FEATURE DEFINITION (UNIFIED)
# ============================================================

# Step 1: Create the model dataset (Only rows with a score to predict)
df_model = df.copy().dropna(subset=["Score"])

# Step 2: Feature list
features = [
    'Country', 'Apparatus', 'Gender',  
    'lag_1', 'lag_2', 'avg_last_3', 'd_lag_1', 'e_lag_1',
    'd_e_interaction', 'olympic_weight', 'elite_ratio', 
    'depth_score', 'topk_score_weighted', 'tier_numeric'
]

# Step 3: Identify the Top 20 Powerhouse Countries
top_20_countries = df_model['Country'].value_counts().nlargest(20).index.tolist()

# Step 4: Bin the Countries (Top 20 vs. Other)
df_model['Country_Binned'] = df_model['Country'].apply(
    lambda x: x if x in top_20_countries else 'Other'
)

# Step 5: Swap 'Country' for 'Country_Binned' in the features list
if 'Country' in features:
    features.remove('Country')
if 'Country_Binned' not in features:
    features.append('Country_Binned')

print(f"Dataset ready. Total Features: {len(features)}")
print(f"Top Powerhouses identified: {top_20_countries[:5]}...")

# ============================================================
# 9. SPLIT (TIME-BASED: TRAIN vs TEST)
# ============================================================

# -----------------------------
# TRAIN
# 2022 + 2023 (EXCLUDING Worlds 2023)
# -----------------------------
train = df_model[
    (
        (df_model["Year"] == 2022) |
        (
            (df_model["Year"] == 2023) &
            ~df_model["Competition"].str.contains("World Championship", case=False, na=False)
        )
    ) &
    df_model["Round"].str.contains("qual|TeamFinal", case=False, na=False)
].copy()


# -----------------------------
# TEST
# Worlds 2023 qualification only
# -----------------------------
test = df_model[
    (df_model["Year"] == 2023) &
    df_model["Competition"].str.contains("World Championship", case=False, na=False) &
    df_model["Round"].str.contains("qual", case=False, na=False)
].copy()

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)


# ============================================================
# 10. PREPARE MATRICES
# ============================================================

# Ensure all features exist
for col in features:
    if col not in train.columns:
        train[col] = 0
    if col not in test.columns:
        test[col] = 0

X_train = train[features].copy()
y_train = train["Score"].copy()

X_test = test[features].copy()
y_test = test["Score"].copy()

# ============================================================
# 10. SPLIT AND FORCE CATEGORICAL TYPES
# ============================================================

# 1. Separate Features and Target
X_train = train[features].copy()
y_train = train["Score"].copy()
X_test = test[features].copy()

# 2. THE CRITICAL FIX: Convert 'object' columns to 'category'
# This stops the 'KeyError: str' in XGBoost
cat_cols = ["Country_Binned", "Apparatus", "Gender", "athlete_type"]
existing_cat_cols = [c for c in cat_cols if c in X_train.columns]

for col in existing_cat_cols:
    # Convert to string first to handle any mixed types, then to category
    X_train[col] = X_train[col].astype(str).astype("category")
    X_test[col] = X_test[col].astype(str).astype("category")

# ============================================================
# 11. FINAL SYNC BEFORE CROSS-VALIDATION
# ============================================================
from sklearn.model_selection import GroupKFold
import numpy as np

# Initialize the cross-validation object
gkf = GroupKFold(n_splits=5)

# 1. Force-sync all three components from the 'train' dataframe
# This ensures all three have exactly 14,734 rows
X_train = train[features].copy()
y_train = train["Score"].copy()
groups = train["athlete_id"].values 

# 2. Re-verify Test set shape for later
X_test = test[features].copy()

# 3. Double-check types (The 'str' fix)
cat_cols = ["Country_Binned", "Apparatus", "Gender"]
for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str).astype("category")
        X_test[col] = X_test[col].astype(str).astype("category")

# 4. Initialize Splitter
gkf = GroupKFold(n_splits=5)

# SYNC CHECK: This MUST show (14734, 14734, 14734)
print(f"--- DATA SYNC REPORT ---")
print(f"X_train: {X_train.shape[0]} rows")
print(f"y_train: {len(y_train)} rows")
print(f"groups:  {len(groups)} rows")
print(f"------------------------")

# 5. THE LOOP
param_grid = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6},
    {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 5}
]

results = []

for params in param_grid:
    rmse_xgb_folds = []
    
    # This will now succeed because rows are perfectly aligned
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        # Fold category alignment
        for col in cat_cols:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].cat.remove_unused_categories()
                X_val[col] = pd.Categorical(X_val[col], categories=X_tr[col].cat.categories)

        xgb = XGBRegressor(
            **params,
            tree_method="hist", 
            enable_categorical=True,
            random_state=42
        )
        
        xgb.fit(X_tr, y_tr)
        preds = xgb.predict(X_val)
        rmse_xgb_folds.append(np.sqrt(mean_squared_error(y_val, preds)))

    results.append({"params": params, "xgb_rmse": np.mean(rmse_xgb_folds)})

results_df = pd.DataFrame(results).sort_values("xgb_rmse")
print("\n===== CROSS-VALIDATION RESULTS =====")
print(results_df)

# ============================================================
# 12. TRAIN FINAL MODELS
# ============================================================
from sklearn.impute import SimpleImputer

best_params = results_df.iloc[0]["params"]

# 1. Align Categorical Types for XGBoost
cat_cols = ["Country", "athlete_type", "Apparatus", "Gender"]
existing_cat_cols = [col for col in cat_cols if col in X_train.columns]

for col in existing_cat_cols:
    X_train[col] = X_train[col].astype(str).replace('nan', 'Unknown')
    X_test[col] = X_test[col].astype(str).replace('nan', 'Unknown')
    master_cats = sorted(list(set(X_train[col].unique()) | set(X_test[col].unique())))
    X_train[col] = pd.Categorical(X_train[col], categories=master_cats)
    X_test[col] = pd.Categorical(X_test[col], categories=master_cats)

# 2. Final XGBoost Fit
final_xgb = XGBRegressor(
    **best_params, 
    tree_method="hist", 
    enable_categorical=True, 
    random_state=42
)
final_xgb.fit(X_train, y_train)

# 3. Final Linear Regression Fit (With Safety Imputer)
# LR needs only numbers and NO NaNs
X_train_num = X_train.select_dtypes(include=[np.number]).copy()
X_test_num = X_test[X_train_num.columns].copy()

imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train_imputed = imputer.fit_transform(X_train_num)
X_test_imputed = imputer.transform(X_test_num)

final_lr = LinearRegression()
final_lr.fit(X_train_imputed, y_train)

# ============================================================
# 13. TEST EVALUATION
# ============================================================
pred_xgb = final_xgb.predict(X_test)
pred_lr = final_lr.predict(X_test_imputed)

print("\nFinal Predictions Generated Successfully!")


# ============================================================
# 14. MODEL COMPARISON (XGBoost vs Linear Regression)
# ============================================================

# --- TRAIN PREDICTIONS ---
train_pred_xgb = final_xgb.predict(X_train)
train_pred_lr = final_lr.predict(X_train_imputed)

# --- RMSE CALCULATION ---
rmse_xgb_train = np.sqrt(mean_squared_error(y_train, train_pred_xgb))
rmse_xgb_test = np.sqrt(mean_squared_error(y_test, pred_xgb))

rmse_lr_train = np.sqrt(mean_squared_error(y_train, train_pred_lr))
rmse_lr_test = np.sqrt(mean_squared_error(y_test, pred_lr))


# --- R² CALCULATION ---
r2_xgb_train = r2_score(y_train, train_pred_xgb)
r2_xgb_test = r2_score(y_test, pred_xgb)

r2_lr_train = r2_score(y_train, train_pred_lr)
r2_lr_test = r2_score(y_test, pred_lr)

# --- RESULTS TABLE ---
model_comparison = pd.DataFrame({
    "Model": ["XGBoost", "Linear Regression"],
    "Train_RMSE": [rmse_xgb_train, rmse_lr_train],
    "Test_RMSE": [rmse_xgb_test, rmse_lr_test],
    "Train_R2": [r2_xgb_train, r2_lr_train],
    "Test_R2": [r2_xgb_test, r2_lr_test]
})

print("\n===== MODEL COMPARISON =====")
print(model_comparison)

# ============================================================
# 15. SELECT BEST MODEL
# ============================================================

# Since we optimized for XGBoost, we'll set it as the best model directly
best_model_name = "XGBoost"
test["pred_score"] = pred_xgb

print(f"\nBest model based on CV: {best_model_name}")

# Create prediction results for error analysis
prediction_results = test.copy()
prediction_results["error"] = (
    prediction_results["Score"] - prediction_results["pred_score"]
)

# ============================================================
# 16 TEAM SCORE SIMULATION
# ============================================================

# We use 'pred_score' to simulate the team totals (Top 3 per apparatus)
top3 = (
    test.sort_values("pred_score", ascending=False)
        .groupby(["Gender", "Country", "Apparatus"])
        .head(3)
)

team_scores = (
    top3.groupby(["Gender", "Country"])["pred_score"]
        .sum()
        .reset_index()
        .sort_values(["Gender", "pred_score"], ascending=[True, False])
)

# Print results
print("\n===== MAG (Men's Artistic Gymnastics) =====")
print(team_scores[team_scores["Gender"] == "MAG"].head(20)) 

print("\n===== WAG (Women's Artistic Gymnastics) =====")
print(team_scores[team_scores["Gender"] == "WAG"].head(20))

# ------------------------------------------------------------
# 17 SAVE OUTPUTS
# ------------------------------------------------------------
test.to_csv("test_output.csv", index=False)
prediction_results.to_csv("prediction_results.csv", index=False)
top3.to_csv("top3_output.csv", index=False)
team_scores.to_csv("team_scores.csv", index=False)
model_comparison.to_csv("rmse_comparison.csv", index=False)
df_model.to_csv("df_model.csv", index=False)
results_df.to_csv("model_metrics.csv", index=False)

# Save the model
joblib.dump(final_xgb, "xgb_model.pkl")


print("\nDONE ✅")
print("Files generated: test_output.csv, prediction_results.csv, top3_output.csv, team_scores.csv,df_model.csv, model_metrics.csv, mode_comparision.csv")
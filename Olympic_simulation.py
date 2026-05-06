
# ============================================================
# OLYMPIC Ranking (TEAM SELECTION + CONSTRAINTS)
# ============================================================

import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================
test = pd.read_csv("test_output.csv")

print("\nLoaded test data:", test.shape)

# ============================================================
# 2. TEAM SIMULATION (Top 4 → Best 3)
# ============================================================

top4 = (
    test.sort_values("pred_score", ascending=False)
        .groupby(["Gender", "Country", "Apparatus"])
        .head(4)
)

top3 = (
    top4.sort_values("pred_score", ascending=False)
        .groupby(["Gender", "Country", "Apparatus"])
        .head(3)
)

team_scores = (
    top3.groupby(["Gender", "Country"])["pred_score"]
        .sum()
        .reset_index()
        .rename(columns={"pred_score": "team_score_pred"})
)

# ============================================================
# 3. RANKING
# ============================================================

def print_ranking(df, gender, title, score_col):
    df_g = (
        df[df["Gender"] == gender]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    df_g.index = df_g.index + 1  # ranking starts at 1
    
    print(f"\n===== {title} =====")
    print(df_g[["Country", score_col]])
    
print_ranking(team_scores, "MAG", "MAG Predicted Ranking", "team_score_pred")
print_ranking(team_scores, "WAG", "WAG Predicted Ranking", "team_score_pred")

# ============================================================
# ACTUAL TEAM FINAL (REAL Worlds 2023)
# ============================================================

df = pd.read_csv(
    "/Users/teresalaramoran/projects/MachineLearningGym/olympic-gymnastics-team-prediction/df_model.csv"
)

world_final = df[
    (df["Competition"].str.contains("World Championship", case=False, na=False)) &
    (df["Year"] == 2023) &
    (df["Round"] == "TeamFinal")
].copy()

team_actual = (
    world_final
    .groupby(["Gender", "Country"])["Score"]
    .sum()
    .reset_index()
    .rename(columns={"Score": "team_score_actual"})
)

team_actual["rank_actual"] = (
    team_actual
    .groupby("Gender")["team_score_actual"]
    .rank(ascending=False, method="min")
)

print_ranking(team_actual, "MAG", "MAG Actual Team Final", "team_score_actual")
print_ranking(team_actual, "WAG", "WAG Actual Team Final", "team_score_actual")

# =======================================================================
# 7. COMPARISION PREDICTED VS REAL TEAM FINAL / SPEARMAN RANK CORRELATION
# =======================================================================

# Keep only teams that are in the real final
comparison = team_scores.merge(
   team_actual,
    on=["Gender", "Country"],
    how="inner"
)

comparison["score_diff"] = (
    comparison["team_score_pred"] - comparison["team_score_actual"]
)
comparison["abs_error"] = comparison["score_diff"].abs()

comparison["rank_pred"] = (
    comparison.groupby("Gender")["team_score_pred"]
        .rank(ascending=False, method="min")
)

comparison["rank_actual"] = (
    comparison.groupby("Gender")["team_score_actual"]
        .rank(ascending=False, method="min")
)
# Add rank difference
comparison["rank_diff"] = comparison["rank_pred"] - comparison["rank_actual"]

# Round values for presentation
comparison["team_score_pred"] = comparison["team_score_pred"].round(2)
comparison["team_score_actual"] = comparison["team_score_actual"].round(2)
comparison["score_diff"] = comparison["score_diff"].round(2)
comparison["abs_error"] = comparison["abs_error"].round(2)

print("\n=== FINALISTS COMPARISON ===")
comparison_display = comparison.sort_values(["Gender", "rank_actual"])[
    [
        "Gender", "Country",
        "team_score_pred", "team_score_actual",
        "score_diff", "abs_error",
        "rank_pred", "rank_actual", "rank_diff"
    ]
]

print(comparison_display.to_string(index=False))
print("\n=== SPEARMAN RESULTS ===")

for gender in comparison["Gender"].unique():
    df_g = comparison[comparison["Gender"] == gender]
    
    corr, pval = spearmanr(df_g["rank_pred"], df_g["rank_actual"])
    
    print(f"\n{gender}:")
    print(f"Spearman Correlation: {corr:.3f}")
    print(f"P-value: {pval:.5f}")


# ============================================================
# 7. SAVE
# ============================================================
team_scores.to_csv("team_scores_predicted.csv", index=False)
comparison_export = comparison[
    [
        "Gender", "Country",
        "team_score_pred", "team_score_actual",
        "score_diff", "abs_error",
        "rank_pred", "rank_actual", "rank_diff"
    ]
].sort_values(["Gender", "rank_actual"])

comparison_export.to_csv("team_comparison_final.csv", index=False)

print("\nDONE ✅")
print("\nDONE ✅ Olympic simulation completed")


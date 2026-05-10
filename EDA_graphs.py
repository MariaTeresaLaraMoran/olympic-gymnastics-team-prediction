import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETUP
# ============================================================

output_dir = "outputs/plots_executive_d_e"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("gymnastics_ml_ready.csv")

# ============================================================
# CLEAN COLUMNS
# ============================================================

for col in ["Apparatus", "Gender", "Round", "Country", "Competition"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Apparatus"] = df["Apparatus"].str.upper()
df["Gender"] = df["Gender"].str.upper()
df["Country"] = df["Country"].str.upper()

# ============================================================
# CREATE ATHLETE TYPE (SAME LOGIC AS MODEL)
# ============================================================

athlete_profile = (
    df.groupby(["athlete_id", "Gender"])["Apparatus"]
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
    athlete_profile[["athlete_id", "athlete_type"]],
    on="athlete_id",
    how="left"
)

# ============================================================
# COUNTRY COLOR MAP
# ============================================================

color_map = {
    "USA": "#1f77b4",   # blue
    "JPN": "#d62728",   # red
    "CHN": "#ff9896",   # light red
    "GBR": "#8c2d2d",   # wine
    "ITA": "#2ca02c",   # green
    "ESP": "#9467bd",   # purple
    "UKR": "#ffd700",   # yellow
    "BRA": "#98df8a",   # light green
    "TUR": "#ff7f0e",
    "SUI": "#17becf",
    "KAZ": "#bcbd22",
    "KOR": "#7f7f7f",
    "GER": "#aec7e8",
    "BEL": "#c5b0d5",
    "CAN": "#ffbb78",
    "FRA": "#3F51B5",
    "AUS": "#e377c2",
    "MEX": "#8c564b",
    "ROU": "#bc80bd",
    "TPE": "#ffed6f",
    "AUT": "#6b6ecf",
    "PAN": "#fd8d3c",
    "NED": "#f7b6d2",
    "ROC": "#636363",
    "RUS": "#636363",
}

# ============================================================
# TOKYO TEAMS
# ============================================================

TOKYO_TEAMS = {
    "MAG": ["JPN", "CHN", "ROC", "RUS", "USA", "GBR", "GER", "ITA", "SUI", "NED", "KOR", "UKR", "BRA"],
    "WAG": ["USA", "ROC", "RUS", "CHN", "FRA", "CAN", "NED", "GBR", "ITA", "GER", "BEL", "JPN", "ESP"]
}

# ============================================================
# EXCLUDE DOMESTIC COMPETITIONS (SAFE VERSION)
# ============================================================

EXCLUDE_PATTERNS = [
    # USA domestic
    "core hydration",
    "winter cup",
    "u.s. championship",
    "u.s. classic",
    "american classic",

    # GBR domestic
    "british gymnastics championship",
]

exclude_pattern = "|".join(EXCLUDE_PATTERNS)

# ============================================================
# APPARATUS COLOR MAP
# ============================================================

apparatus_colors = {
    "FX": "#1f77b4",
    "PH": "#ff7f0e",
    "SR": "#2ca02c",
    "VT": "#d62728",
    "PB": "#9467bd",
    "HB": "#8c564b",
    "UB": "#e377c2",
    "BB": "#17becf",
}

mag_order = ["FX", "PH", "SR", "VT", "PB", "HB"]
wag_order = ["VT", "UB", "BB", "FX"]

# ============================================================
# HELPER: FACET SUBPLOTS
# ============================================================

def get_axes_for_apparatus(apparatus_list):
    n_cols = 3
    n_rows = (len(apparatus_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    return fig, axes, n_rows, n_cols

# ============================================================
# 1. QUALIFICATIONS 2022-2023 SCATTER FACET BY APPARATUS
# ============================================================

def plot_qual(df, gender="MAG", save=True):
    data = df.copy()
    data = data[
        (data["Year"].isin([2022, 2023])) &
        (data["Round"].str.contains("qual", case=False, na=False)) &
        (data["Gender"] == gender)
    ].copy()

    top_countries = list(color_map.keys())
    data = data[data["Country"].isin(top_countries)].copy()

    if data.empty:
        print(f"⚠️ No qualification data for {gender}")
        return

    apparatus_list = sorted(data["Apparatus"].dropna().unique())
    fig, axes, _, _ = get_axes_for_apparatus(apparatus_list)

    for i, app in enumerate(apparatus_list):
        ax = axes[i]
        subset = data[data["Apparatus"] == app]

        for country in top_countries:
            sub_c = subset[subset["Country"] == country]
            if sub_c.empty:
                continue

            is_highlight = country in TOKYO_TEAMS.get(gender, [])

            ax.scatter(
                sub_c["D_Score"],
                sub_c["Score"],
                color=color_map.get(country, "#999999"),
                alpha=0.85 if is_highlight else 0.35,
                s=40 if is_highlight else 14,
                edgecolors="black" if is_highlight else None,
                linewidth=0.4 if is_highlight else 0
            )

        ax.set_title(app)
        ax.set_xlabel("D Score")
        ax.set_ylabel("Score")

    for j in range(len(apparatus_list), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{gender} - D Score vs Score (Qualifications 2022–2023)", fontsize=16)

    for country in TOKYO_TEAMS.get(gender, []):
        if country in data["Country"].unique():
            plt.plot([], [], color=color_map.get(country, "#999999"), label=country, linewidth=3)

    plt.legend(
        title="Tokyo Teams",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if save:
        filename = f"{output_dir}/{gender.lower()}_qual.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print("Saved:", filename)

    plt.show()
    plt.close()

# ============================================================
# 2. TOKYO SCATTER FACET BY APPARATUS
# ============================================================

def plot_tokyo(df, gender="MAG", save=True):
    data = df.copy()
    data = data[
        (data["Year"].isin([2020, 2021])) &
        (data["Round"].str.contains("Team", case=False, na=False)) &
        (data["Gender"] == gender)
    ].copy()

    top_countries = list(color_map.keys())
    data = data[data["Country"].isin(top_countries)].copy()

    if data.empty:
        print(f"⚠️ No Tokyo data for {gender}")
        return

    apparatus_list = sorted(data["Apparatus"].dropna().unique())
    fig, axes, _, _ = get_axes_for_apparatus(apparatus_list)

    for i, app in enumerate(apparatus_list):
        ax = axes[i]
        subset = data[data["Apparatus"] == app]

        for country in sorted(subset["Country"].unique()):
            sub_c = subset[subset["Country"] == country]
            if sub_c.empty:
                continue

            is_highlight = country in TOKYO_TEAMS.get(gender, [])

            alpha = 0.85 if is_highlight else 0.35
            size = 42 if is_highlight else 14
            edge = "black" if is_highlight else None
            lw = 0.4 if is_highlight else 0

            for round_name, marker in [("TEAMQUAL", "o"), ("TEAMFINAL", "x")]:
                sub_r = sub_c[sub_c["Round"].str.upper() == round_name]
                if sub_r.empty:
                    continue

                ax.scatter(
                    sub_r["D_Score"],
                    sub_r["Score"],
                    color=color_map.get(country, "#999999"),
                    alpha=alpha,
                    s=size,
                    marker=marker,
                    edgecolors=edge,
                    linewidth=lw
                )

        ax.set_title(app)
        ax.set_xlabel("D Score")
        ax.set_ylabel("Score")

    for j in range(len(apparatus_list), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{gender} - Tokyo TeamQual vs TeamFinal", fontsize=16)

    for country in TOKYO_TEAMS.get(gender, []):
        if country in data["Country"].unique():
            plt.plot([], [], color=color_map.get(country, "#999999"), label=country, linewidth=3)

    plt.legend(
        title="Tokyo Teams",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if save:
        filename = f"{output_dir}/{gender.lower()}_tokyo.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print("Saved:", filename)

    plt.show()
    plt.close()

# ============================================================
# 3. AA vs specialits
# ========================================================


def plot_aa_vs_specialist_clean(df, gender="MAG", save=True):

    # ============================================================
    # FILTER: REMOVE DOMESTIC
    # ============================================================
    
    df = df[
    ~df["Competition"].str.contains("core hydration", case=False, na=False)
    ]
    
    df_int = df[
        ~df["Competition"].astype(str).str.contains(
            exclude_pattern,
            case=False,
            na=False
        )
    ].copy()

    data = df_int[df_int["Gender"] == gender].copy()

    if data.empty:
        print(f"⚠️ No data for {gender}")
        return

    # ============================================================
    # UNIQUE ATHLETES
    # ============================================================

    athletes = (
        data[["athlete_id", "Country", "athlete_type"]]
        .drop_duplicates()
    )

    # ============================================================
    # PIVOT
    # ============================================================

    pivot = (
        athletes
        .groupby(["Country", "athlete_type"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["AA", "Specialist"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # ============================================================
    # TOP 12 + MEX + CHN (CORRECT ORDER)
    # ============================================================

    pivot["Total"] = pivot.sum(axis=1)

    # Sort biggest first
    pivot_sorted = pivot.sort_values("Total", ascending=False)

    # Top 12
    top12 = pivot_sorted.head(12)

    # Countries to force include
    force_include = ["MEX", "CHN"]

    # Add missing ones
    extras = []
    for c in force_include:
        if c in pivot_sorted.index and c not in top12.index:
            extras.append(pivot_sorted.loc[[c]])

    # Combine
    if extras:
        pivot = pd.concat([top12] + extras)
    else:
        pivot = top12.copy()

    # Remove duplicates (safety)
    pivot = pivot[~pivot.index.duplicated(keep="first")]

    # Sort for barh (small → large)
    pivot = pivot.sort_values("Total", ascending=True)

    # Drop helper column
    pivot = pivot.drop(columns="Total")

    # ============================================================
    # PLOT
    # ============================================================

    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10,6))

    pivot.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=colors
    )

    # ============================================================
    # CLEAN STYLE
    # ============================================================

    ax.set_xlabel("Number of Athletes")
    ax.set_ylabel("")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ============================================================
    # LABELS INSIDE BARS
    # ============================================================

    for i, (aa, sp) in enumerate(zip(pivot["AA"], pivot["Specialist"])):
        if aa > 0:
            ax.text(aa/2, i, str(int(aa)), va='center', ha='center', color='white', fontsize=8)
        if sp > 0:
            ax.text(aa + sp/2, i, str(int(sp)), va='center', ha='center', color='white', fontsize=8)

    # ============================================================
    # LEGEND
    # ============================================================

    ax.legend(loc="lower right", frameon=False)

    plt.title(f"{gender} - AA vs Specialist (Top 12)")
    plt.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    if save:
        filename = f"{output_dir}/{gender.lower()}_aa_vs_specialist_clean.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print("Saved:", filename)

    plt.show()
    plt.close()

def plot_country_apparatus(df, gender="MAG", save=True):

    # ============================================================
    # FILTER (INTERNATIONAL + QUAL)
    # ============================================================

    df_int = df[
        ~df["Competition"].str.contains(exclude_pattern, case=False, na=False)
    ].copy()

    data = df_int[
        (df_int["Gender"] == gender) &
        (df_int["Round"].str.contains("qual", case=False, na=False))
    ].copy()

    if data.empty:
        print(f"⚠️ No qualification international data for {gender}")
        return

    # ============================================================
    # PIVOT
    # ============================================================

    pivot_app = (
        data.pivot_table(
            index="Country",
            columns="Apparatus",
            values="Score",
            aggfunc="count",
            fill_value=0
        )
    )

    # Order apparatus
    if gender == "MAG":
        pivot_app = pivot_app.reindex([c for c in mag_order if c in pivot_app.columns], axis=1)
    else:
        pivot_app = pivot_app.reindex([c for c in wag_order if c in pivot_app.columns], axis=1)

    # ============================================================
    # TOP 12 + MEX + CHN
    # ============================================================

    pivot_app["Total"] = pivot_app.sum(axis=1)

    pivot_sorted = pivot_app.sort_values("Total", ascending=False)

    top12 = pivot_sorted.head(12)

    force_include = ["MEX", "CHN"]
    extras = []

    for c in force_include:
        if c in pivot_sorted.index and c not in top12.index:
            extras.append(pivot_sorted.loc[[c]])

    if extras:
        pivot_app = pd.concat([top12] + extras)
    else:
        pivot_app = top12.copy()

    pivot_app = pivot_app[~pivot_app.index.duplicated(keep="first")]

    # Sort for barh
    pivot_app = pivot_app.sort_values("Total", ascending=True)

    pivot_app = pivot_app.drop(columns="Total")

    # ============================================================
    # COLORS
    # ============================================================

    colors = [apparatus_colors.get(app, "#cccccc") for app in pivot_app.columns]

    # ============================================================
    # PLOT
    # ============================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot_app.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=colors
    )

    # ============================================================
    # CLEAN STYLE
    # ============================================================

    ax.set_title(f"{gender} - Routines by Country & Apparatus (Qualifications)")
    ax.set_xlabel("Number of Routines")
    ax.set_ylabel("")

    # Remove only unnecessary spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Clean ticks (keep readable)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=10)

    # ============================================================
    # LEGEND (clean + consistent)
    # ============================================================

    ax.legend(
        title="",
        loc="lower right",
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    if save:
        filename = f"{output_dir}/{gender.lower()}_country_apparatus.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print("Saved:", filename)

    plt.show()
    plt.close()
# ============================================================
# RUN ALL
# ============================================================

for gender in ["MAG", "WAG"]:
    plot_qual(df, gender)
    plot_tokyo(df, gender)
    plot_country_apparatus(df, gender)
    plot_aa_vs_specialist_clean(df, gender)
    
    

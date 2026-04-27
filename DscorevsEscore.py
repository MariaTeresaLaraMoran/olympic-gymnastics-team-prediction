# D score vs E score plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Output folder
output_dir = "outputs/plots_executive_d_e"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("gymnastics_ml_ready.csv")

print("Shape:", df.shape)
print(df.head())

# ------------------------------------------------------------
# FIX COLUMNS + GENDER FORMAT
# ------------------------------------------------------------

# Clean column names
df.columns = df.columns.str.strip()

# Standardize Gender
df["Gender"] = df["Gender"].replace({
    "M": "MAG",
    "W": "WAG"
})

# Plot
def plot_d_e_facet_by_gender(df, gender):

    # Filter gender
    data = df[df["Gender"] == gender].copy()
    data = data.dropna(subset=["D_Score", "E_Score", "Score"])

    apparatus_list = sorted(data["Apparatus"].unique())

    # Layout
    n = len(apparatus_list)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    sc = None  # 👈 important

    for i, app in enumerate(apparatus_list):
        ax = axes[i]

        subset = data[data["Apparatus"] == app]

        sc = ax.scatter(
            subset["D_Score"],
            subset["E_Score"],
            c=subset["Score"],
            alpha=0.6,
            s=10
        )

        ax.set_title(app, fontsize=12)

        # Cleaner labels
        if i % cols == 0:
            ax.set_ylabel("E-score")

        if i >= (rows - 1) * cols:
            ax.set_xlabel("D-score")

        ax.grid(True, linestyle="--", alpha=0.3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    #  Create colorbar FIRST
    if sc is not None:
        
        # ------------------------------------------------------------
        # CLEAN TOP COLORBAR (EXECUTIVE STYLE)
        # ------------------------------------------------------------
        cbar_ax = fig.add_axes([0.25, 0.93, 0.5, 0.02])  
        # [left, bottom, width, height]

        cbar = fig.colorbar(
            sc,
            cax=cbar_ax,
            orientation="horizontal"
            )

        cbar.set_label("Total Score", fontsize=10)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_coords(0, 2.2)

        fig.suptitle(f"D-Score vs E-Score — {gender}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    filename = f"{output_dir}/facet_d_e_{gender}.png"
    plt.savefig(filename, bbox_inches="tight")

    plt.show()
    
# Run for gender
plot_d_e_facet_by_gender(df, "MAG")
plot_d_e_facet_by_gender(df, "WAG")
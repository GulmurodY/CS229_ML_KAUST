import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../output_csv/wine_mse_comparison.csv")
metrics = dict(zip(df["Metric"], df["Value"]))

bases  = ["polynomial", "gaussian", "sigmoidal"]
labels = ["Polynomial", "Gaussian", "Sigmoidal"]
train_mses = [metrics[f"wine_{b}_train_MSE"] for b in bases]
val_mses   = [metrics[f"wine_{b}_val_MSE"]   for b in bases]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, train_mses, width, label="Train MSE", color="steelblue")
bars2 = ax.bar(x + width/2, val_mses,   width, label="Val MSE",   color="tomato")

ax.set_xlabel("Basis Function")
ax.set_ylabel("MSE")
ax.set_title("Bayesian Linear Regression — Wine Dataset\nMSE by Basis Function")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("../output_plots/basis_comparison_plot.png", dpi=150)
plt.show()
print("Saved: ../output_plots/basis_comparison_plot.png")

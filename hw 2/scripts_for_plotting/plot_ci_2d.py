import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../output_csv/wine_polynomial_predictions.csv")
df = df.sort_values("predicted_mean").reset_index(drop=True)

mean   = df["predicted_mean"].values
low    = df["lower_95CI"].values
high   = df["upper_95CI"].values
actual = df["actual"].values
ci_half = (high - low) / 2

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

idx = np.arange(len(mean))
axes[0].fill_between(idx, low, high, alpha=0.3, color="steelblue", label="95% CI")
axes[0].plot(idx, mean, color="steelblue", linewidth=1.2, label="Predicted mean")
axes[0].scatter(idx, actual, s=6, color="tomato", alpha=0.5, zorder=5, label="Actual")
axes[0].set_xlabel("Sample index (sorted by predicted mean)")
axes[0].set_ylabel("Wine quality")
axes[0].set_title("95% Confidence Intervals — Wine Dataset\n(11 features, polynomial degree 2 — 2D+ data)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

step = max(1, len(mean) // 150)
idx_sub = np.arange(0, len(mean), step)
axes[1].errorbar(actual[idx_sub], mean[idx_sub],
                 yerr=ci_half[idx_sub],
                 fmt="o", markersize=3, color="steelblue",
                 ecolor="steelblue", elinewidth=0.6, alpha=0.6,
                 label="Prediction ± 95% CI")
lims = [min(actual.min(), mean.min()) - 0.2,
        max(actual.max(), mean.max()) + 0.2]
axes[1].plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("Actual wine quality")
axes[1].set_ylabel("Predicted wine quality")
axes[1].set_title("Predicted vs Actual with 95% CI\nWine Dataset (2D+ data)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../output_plots/ci_2d_plot.png", dpi=150)
plt.show()
print("Saved: ../output_plots/ci_2d_plot.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../output_csv/synthetic_predictions.csv")
df = df.sort_values("x0")

x      = df["x0"].values
actual = df["actual"].values
mean   = df["predicted_mean"].values
lo     = df["lower_95CI"].values
hi     = df["upper_95CI"].values

x_true = np.linspace(0, 1, 300)
y_true = np.sin(2 * np.pi * x_true)

fig, ax = plt.subplots(figsize=(9, 5))

ax.fill_between(x, lo, hi, alpha=0.25, color="steelblue", label="95% CI")
ax.plot(x, mean, color="steelblue", linewidth=2, label="Predicted mean")
ax.plot(x_true, y_true, "k--", linewidth=1.5, label="True sin(2πx)")
ax.scatter(x, actual, s=25, color="tomato", zorder=5, label="Observations")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Bayesian Linear Regression — Synthetic 1D Data\n(Polynomial basis, degree 9)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../output_plots/synthetic_ci_plot.png", dpi=150)
plt.show()
print("Saved: ../output_plots/synthetic_ci_plot.png")

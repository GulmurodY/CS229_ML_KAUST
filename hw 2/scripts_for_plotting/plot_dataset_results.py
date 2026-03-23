import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parity_plot(ax, actual, mean, lo, hi, title, xlabel="Actual", ylabel="Predicted"):
    ci_half = (hi - lo) / 2
    step = max(1, len(actual) // 200)
    s = slice(None, None, step)
    ax.errorbar(actual[s], mean[s], yerr=ci_half[s],
                fmt="o", markersize=3, color="steelblue",
                ecolor="steelblue", elinewidth=0.5, alpha=0.55,
                label="Prediction ± 95% CI")
    lim = [min(actual.min(), mean.min()) - 0.05 * abs(actual.min()),
           max(actual.max(), mean.max()) + 0.05 * abs(actual.max())]
    ax.plot(lim, lim, "k--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


def residual_plot(ax, actual, mean, title):
    residuals = actual - mean
    ax.scatter(mean, residuals, s=6, alpha=0.4, color="tomato")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted mean"); ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(title); ax.grid(True, alpha=0.3)


df = pd.read_csv("../output_csv/synthetic_predictions.csv")
df = df.sort_values("x0").reset_index(drop=True)

x      = df["x0"].values
actual = df["actual"].values
mean   = df["predicted_mean"].values
lo     = df["lower_95CI"].values
hi     = df["upper_95CI"].values
x_true = np.linspace(0, 1, 300)
y_true = np.sin(2 * np.pi * x_true)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].fill_between(x, lo, hi, alpha=0.25, color="steelblue", label="95% CI")
axes[0].plot(x, mean, color="steelblue", linewidth=2, label="Predicted mean")
axes[0].plot(x_true, y_true, "k--", linewidth=1.5, label="True sin(2πx)")
axes[0].scatter(x, actual, s=25, color="tomato", zorder=5, label="Observations")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("Bayesian LR — Synthetic 1D\n(Polynomial degree 9, 95% CI)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

residual_plot(axes[1], actual, mean, "Residuals — Synthetic 1D")

plt.suptitle("Synthetic Dataset Results", fontweight="bold")
plt.tight_layout()
plt.savefig("../output_plots/results_synthetic.png", dpi=150)
plt.show()
print("Saved: ../output_plots/results_synthetic.png")


bases = [("polynomial", "Polynomial deg-2"), ("gaussian", "Gaussian"), ("sigmoidal", "Sigmoidal")]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, (bname, blabel) in enumerate(bases):
    df = pd.read_csv(f"../output_csv/wine_{bname}_predictions.csv")
    actual = df["actual"].values
    mean   = df["predicted_mean"].values
    lo     = df["lower_95CI"].values
    hi     = df["upper_95CI"].values

    parity_plot(axes[0, col], actual, mean, lo, hi,
                f"Wine — {blabel}\nPredicted vs Actual (95% CI)",
                xlabel="Actual quality", ylabel="Predicted quality")

    residual_plot(axes[1, col], actual, mean,
                  f"Wine — {blabel}\nResiduals")

plt.suptitle("Wine Dataset Results (11 features — 2D+ data)", fontweight="bold")
plt.tight_layout()
plt.savefig("../output_plots/results_wine.png", dpi=150)
plt.show()
print("Saved: ../output_plots/results_wine.png")


df = pd.read_csv("../output_csv/automobile_predictions.csv")
actual = df["actual"].values
mean   = df["predicted_mean"].values
lo     = df["lower_95CI"].values
hi     = df["upper_95CI"].values

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

parity_plot(axes[0], actual, mean, lo, hi,
            "Automobile — Polynomial deg-2\nPredicted vs Actual (95% CI)",
            xlabel="Actual price ($)", ylabel="Predicted price ($)")

residual_plot(axes[1], actual, mean, "Automobile — Residuals")

plt.suptitle("Automobile Dataset Results (15 features — 2D+ data)", fontweight="bold")
plt.tight_layout()
plt.savefig("../output_plots/results_automobile.png", dpi=150)
plt.show()
print("Saved: ../output_plots/results_automobile.png")

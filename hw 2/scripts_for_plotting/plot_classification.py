import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DIR = "../output_csv"
OUT_DIR = "../output_plots"


def plot_loss_curve(dataset: str) -> None:
    df = pd.read_csv(f"{CSV_DIR}/classification_loss_{dataset}.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["loss"], color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title(f"LogReg training loss — {dataset}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/classification_loss_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_confusion(dataset: str, model: str, ax) -> None:
    df = pd.read_csv(f"{CSV_DIR}/classification_confusion_{dataset}_{model}.csv")
    cm = np.array([[df["TN"].iloc[0], df["FP"].iloc[0]],
                   [df["FN"].iloc[0], df["TP"].iloc[0]]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    acc = df["accuracy"].iloc[0]
    f1  = df["f1"].iloc[0]
    ax.set_title(f"{model.upper()} — acc={acc:.3f}  f1={f1:.3f}")


def plot_confusion_pair(dataset: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_confusion(dataset, "logreg", axes[0])
    plot_confusion(dataset, "gda",    axes[1])
    fig.suptitle(f"Confusion matrices — {dataset}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/classification_confusion_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_decision_boundary() -> None:
    data = pd.read_csv(f"{CSV_DIR}/classification_synthetic_data.csv")
    grid_lr  = pd.read_csv(f"{CSV_DIR}/classification_synthetic_grid_logreg.csv")
    grid_gda = pd.read_csv(f"{CSV_DIR}/classification_synthetic_grid_gda.csv")

    x1_vals = np.sort(grid_lr["x1"].unique())
    x2_vals = np.sort(grid_lr["x2"].unique())
    n1, n2 = len(x1_vals), len(x2_vals)

    X1 = grid_lr["x1"].to_numpy().reshape(n1, n2)
    X2 = grid_lr["x2"].to_numpy().reshape(n1, n2)
    P_lr  = grid_lr ["prob"].to_numpy().reshape(n1, n2)
    P_gda = grid_gda["prob"].to_numpy().reshape(n1, n2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    levels = np.linspace(0.0, 1.0, 11)

    for ax, P, name in [(axes[0], P_lr, "LogReg"),
                        (axes[1], P_gda, "GDA")]:
        cf = ax.contourf(X1, X2, P, levels=levels, cmap="RdBu_r", alpha=0.7)
        ax.contour(X1, X2, P, levels=[0.5], colors="black", linewidths=1.5)
        pos = data[data["y"] == 1]
        neg = data[data["y"] == 0]
        ax.scatter(neg["x1"], neg["x2"], s=14, c="#08306b", edgecolor="white",
                   linewidths=0.4, label="class 0")
        ax.scatter(pos["x1"], pos["x2"], s=14, c="#67000d", edgecolor="white",
                   linewidths=0.4, label="class 1")
        ax.set_title(f"{name} — decision boundary (p=0.5 in black)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.8, label="P(y=1 | x)")
    fig.suptitle("Synthetic 2D — LogReg vs GDA")
    fig.savefig(f"{OUT_DIR}/classification_decision_boundary.png", dpi=150)
    plt.close(fig)


def plot_summary_bars() -> None:
    df = pd.read_csv(f"{CSV_DIR}/classification_summary.csv")
    metrics = ["accuracy", "precision", "recall", "f1"]
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        x = np.arange(len(metrics))
        width = 0.35
        for i, model in enumerate(["logreg", "gda"]):
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            vals = [row[m].iloc[0] for m in metrics]
            ax.bar(x + (i - 0.5) * width, vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ds)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Classification metrics summary")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/classification_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_loss_curve("titanic")
    plot_loss_curve("iris_setosa_vs_rest")
    plot_confusion_pair("titanic")
    plot_confusion_pair("iris_setosa_vs_rest")
    plot_decision_boundary()
    plot_summary_bars()
    print("Plots written to", OUT_DIR)

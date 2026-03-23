import matplotlib.pyplot as plt


def parse_posterior_mean(filepath):
    weights = {}
    reading = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == "posterior_mean":
                reading = True
                continue
            if line == "" or line == "posterior_covariance":
                reading = False
            if reading and line:
                key, val = line.split(",")
                weights[key] = float(val)
    return list(weights.keys()), list(weights.values())


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

labels, values = parse_posterior_mean("../output_csv/synthetic_posterior.csv")
colors = ["steelblue" if v >= 0 else "tomato" for v in values]
axes[0].bar(range(len(labels)), values, color=colors)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
axes[0].set_xlabel("Weight index")
axes[0].set_ylabel("m_N value")
axes[0].set_title("Posterior Mean Weights m_N\nSynthetic 1D data (polynomial degree 9)")
axes[0].grid(True, axis="y", alpha=0.3)

labels2, values2 = parse_posterior_mean("../output_csv/wine_polynomial_posterior.csv")
colors2 = ["steelblue" if v >= 0 else "tomato" for v in values2]
axes[1].bar(range(len(labels2)), values2, color=colors2)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xticks(range(len(labels2)))
axes[1].set_xticklabels(labels2, rotation=60, ha="right", fontsize=7)
axes[1].set_xlabel("Weight index")
axes[1].set_ylabel("m_N value")
axes[1].set_title("Posterior Mean Weights m_N\nWine dataset — 11 features, polynomial degree 2 (2D+ data)")
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../output_plots/posterior_weights_plot.png", dpi=150)
plt.show()
print("Saved: ../output_plots/posterior_weights_plot.png")

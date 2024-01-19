# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import balanced_accuracy_score
plt.rcParams['legend.title_fontsize'] = 8
# %%
fnames = list(Path("results/valid_params/").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["bal"] = df.apply(lambda x: balanced_accuracy_score(x.y_target, x.y_pred), axis=1)

fnames = list(Path("results/LODO/").glob("*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_baseline["bal"] = df_baseline.apply(lambda x: balanced_accuracy_score(x.y_target, x.y_pred), axis=1)
df_baseline = df_baseline.groupby(["dataset_t", "method"]).mean().reset_index()
# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True) #,sharey=True)
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]
df_plot = df.groupby(["dataset_t", "filter_size", "reg"]).mean().reset_index()
for i, ax in enumerate(axes.flatten()):
    sns.lineplot(
        data=df_plot.query(f"dataset_t == '{dataset[i]}' & reg < 0.01"),
        x="filter_size",
        y="bal",
        hue="reg",
        palette="colorblind",
        ax=ax,
        marker="o",
        markersize=8,
        linewidth=2,
        alpha=0.8,
        legend=False if i != 1 else True,
    )
    # plot baseline horizontal line
    if i == 1:
        ax.legend(fontsize="8")#bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        ax.get_legend().set_title("Regularization",)
    line1 = ax.axhline(
        df_baseline.query(f"dataset_t == '{dataset[i]}' & method == 'raw'")["bal"].values[0],
        linestyle="--",
        color="k",
        label="No Aligment",
        alpha=0.8,
    )
    line2 = ax.axhline(
        df_baseline.query(f"dataset_t == '{dataset[i]}' & method == 'riemann'")["bal"].values[0],
        linestyle=":",
        color="k",
        label="Pre-Whitening",
        alpha=0.8,
    )
    line3 = ax.axhline(
        df_baseline.query(f"dataset_t == '{dataset[i]}' & method == 'temp'")["bal"].values[0],
        linestyle="-.",
        color="k",
        label="Temporal CMMN",
        alpha=0.8,
    )
    if i == 2:
        first_legend = ax.legend(handles=[line1, line2, line3], fontsize=8) #, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.get_legend().set_title("Chambon")
        # plt.gca().add_artist(first_legend)

    ax.set_xscale("log")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048,])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    ax.set_title("Dataset Target: " + dataset[i])
    ax.set_xlabel("Filter size")
    ax.set_ylabel("Accuracy")
    # ax.set_ylim(0.55, 0.8)
fig.savefig("results/plots/valid_params.pdf", bbox_inches="tight")
# %%

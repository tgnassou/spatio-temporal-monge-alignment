# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import text

from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon

# %% Plot Spatio-temp vs RA and raw
fnames = list(Path("results/LODO").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df = df.query(
    "filter == True"
    "& n_epochs == 200"
    "& archi == 'ShallowNet'"
    "& num_iter == 10"
)
# df = df.query("method != 'spatiotemp' | num_iter == 200")
# %%
pairs = [
    ("raw", "spatiotemp"),
    ("raw", "riemannalignment"),
    ("spatiotemp", "riemannalignment"),
]
fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
axes[1][2].set_visible(False)
dataset = [
    "BNCI2014001", "PhysionetMI", "Weibo2014",
    "Cho2017", "Schirrmeister2017"
]
df_plot = df.groupby(
    ["subject_test", "dataset_target", "method"]
).mean().reset_index()
df_plot = df_plot.query("method in ['raw', 'riemannalignment', 'spatiotemp']")
for i, ax in enumerate(axes.flatten()):
    if i == 5:
        continue
    df_plot_ = df_plot.query(f"dataset_target == '{dataset[i]}'")

    raw = df_plot_.loc[(df_plot_.method == "raw"), "acc"].values
    riemann = df_plot_.loc[(df_plot_.method == "riemannalignment"), "acc"].values
    spatiotemp = df_plot_.loc[(df_plot_.method == "spatiotemp"), "acc"].values

    pvalues = [
        wilcoxon(raw, spatiotemp, alternative="two-sided").pvalue,
        wilcoxon(raw, riemann, alternative="two-sided").pvalue,
        wilcoxon(spatiotemp, riemann, alternative="two-sided").pvalue
    ]

    axis = sns.boxplot(
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="acc",
        y="method",
        orient="h",
        palette="colorblind",
        showfliers=False,
        ax=ax,
    )
    annotator = Annotator(
        ax,
        pairs,
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="acc",
        y="method",
        orient="h",
        palette="colorblind",
        showfliers=False,
    )
    annotator.set_pvalues(pvalues)
    # annotator.annotate()
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    sns.stripplot(
        data=df_plot_,
        x="acc",
        y="method",
        orient="h",
        dodge=False,
        # legend=False,
        ax=ax,
        alpha=0.3,
        size=5,
        color="silver",
        edgecolor="dimgray",
        linewidth=0.1,
    )
    acc = df_plot_.groupby("method").mean().reset_index().acc
    std = df_plot_.groupby("method").std().reset_index().acc
    labels = [
        f"w/o Normalization \n ({acc[0]:.2f} $\pm$ {std[0]:.2f}) ", # noqa
        f"Riemann Alignment \n ({acc[1]:.2f} $\pm$ {std[1]:.2f})", # noqa
        f"Spatio-Temp CMMN  \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})", # noqa
        # f"Temp CMMN  \n ({acc[3]:.2f} $\pm$ {std[3]:.2f})"
    ]
    ax.set_yticklabels(labels)
    ax.set_title(f"Target: {dataset[i]}")
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("results/plots/LODO_bci.pdf", bbox_inches="tight")
# %%
fig, axes = plt.subplots(
    1, 2, figsize=(6, 3),
    sharex=True, sharey=True, layout="constrained"
)
cmap = sns.color_palette("colorblind", as_cmap=True)

ax = axes[0]
df_plot = pd.concat(
    (
        df.query("method == 'raw'")[["acc"]],
        df.query("method == 'spatiotemp'").rename(columns={"acc": "acc_adapted"})),
    axis=1
)
raw = df_plot.acc.values
spatiotemp = df_plot.acc_adapted.values
# len of spatiotemp - raw positive
n = np.sum(spatiotemp - raw > 0)
pvalue = wilcoxon(spatiotemp, raw, alternative="two-sided").pvalue
ax.set_title(f"p-value: {pvalue:.2f}")
df_plot = df_plot[["subject_test", "dataset_target", "acc_adapted", "acc"]]
df_plot["delta"] = df_plot.acc_adapted - df_plot.acc

fig.tight_layout(rect=[0, 0, .9, 1])
sns.scatterplot(
    data=df_plot,
    x="acc",
    y="acc_adapted",
    alpha=0.8,
    linewidth=1,
    ax=ax,
    legend=False,
    hue="dataset_target",
    palette="colorblind"
)
text(
    0.11, 0.23, f"{np.int(np.round(n/len(raw)*100))}%",
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes, size=9
)
text(
    0.23, 0.07, f"{np.int(np.round((1 - n/len(raw))*100))}%",
    horizontalalignment='center', verticalalignment='center',
    transform=ax.transAxes,size=9
)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Acc. with No Align.")
ax.set_ylabel("Acc. with STMA")
ax = axes[1]
df_plot = pd.concat(
    (
        df.query("method == 'riemannalignment'")[["acc"]],
        df.query(
            "method == 'spatiotemp'"
        ).rename(columns={"acc": "acc_adapted"})
    ),
    axis=1
)
riemann = df_plot.acc.values
pvalue = wilcoxon(spatiotemp, riemann, alternative="two-sided").pvalue
n = np.sum(spatiotemp - riemann > 0)
ax.set_title(f"p-value: {pvalue:.2f}")

df_plot = df_plot[["subject_test", "dataset_target", "acc_adapted", "acc"]]
df_plot["delta"] = df_plot.acc_adapted - df_plot.acc

g = sns.scatterplot(
    data=df_plot,
    x="acc",
    y="acc_adapted",
    alpha=0.8,
    linewidth=1,
    ax=ax,
    hue="dataset_target",
    palette="colorblind"
)
text(0.11, 0.23, f"{np.int(np.round(n/len(raw)*100))}%", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=9)
text(0.23, 0.07, f"{np.int(np.round((1 - n/len(raw))*100))}%", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,size=9)
plt.legend(title="Dataset Target", loc="upper left", bbox_to_anchor=(1.1, 0.8), )
ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Acc. with RA")

fig.savefig("results/plots/scatter_bci.pdf", bbox_inches="tight")

# %%
# print ablation study

df_plot = df.query("method in ['temp', 'raw', 'spatio', 'spatiotemp']")

fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
axes[1][2].set_visible(False)
df_plot = df_plot.groupby(
    ["subject_test", "dataset_target", "method"]
).mean().reset_index()
for i, ax in enumerate(axes.flatten()):
    if i == 5:
        continue
    df_plot_ = df_plot.query(f"dataset_target == '{dataset[i]}'")

    raw = df_plot_.loc[(df_plot_.method == "raw"), "acc"].values
    riemann = df_plot_.loc[(df_plot_.method == "riemannalignment"), "acc"].values
    spatiotemp = df_plot_.loc[(df_plot_.method == "spatiotemp"), "acc"].values

    pvalues = [
        wilcoxon(raw, spatiotemp, alternative="two-sided").pvalue,
        wilcoxon(raw, riemann, alternative="two-sided").pvalue,
        wilcoxon(spatiotemp, riemann, alternative="two-sided").pvalue
    ]

    axis = sns.boxplot(
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="acc",
        y="method",
        orient="h",
        palette="colorblind",
        showfliers=False,
        ax=ax,
    )
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    sns.stripplot(
        data=df_plot_,
        x="acc",
        y="method",
        orient="h",
        dodge=False,
        legend=False,
        ax=ax,
        alpha=0.3,
        size=5,
        color="silver",
        edgecolor="dimgray",
        linewidth=0.1,
    )
    acc = df_plot_.groupby("method").mean().reset_index().acc
    std = df_plot_.groupby("method").std().reset_index().acc
    labels = [
        f"w/o Normalization \n ({acc[0]:.2f} $\pm$ {std[0]:.2f}) ", # noqa
        f"Spatio CMMN\n ({acc[1]:.2f} $\pm$ {std[1]:.2f})", # noqa
        f"Spatio-Temp CMMN  \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})", # noqa
        f"Temp CMMN  \n ({acc[3]:.2f} $\pm$ {std[3]:.2f})"
    ]
    ax.set_yticklabels(labels)
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("results/plots/LODO_ablation_bci.pdf", bbox_inches="tight")
    # plt.xlabel("BACC")
# %%
df_plot = df.query("method in ['temp', 'raw', 'spatio', 'spatiotemp']")
df_tab = df_plot[['subject_test', 'dataset_target', 'method', 'acc']]
df_tab = df_tab.groupby(['dataset_target', 'method',]).agg({"acc": ["mean", "std"]})
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.acc['mean']:.2f} $\pm$ {x.acc['std']:.2f}", axis=1 # noqa
)
# pivot for method and dataset_t
df_tab = df_tab.reset_index().pivot(index="method", columns="dataset_target", values="mean_std")
print(df_tab.to_latex(escape=False))
# %%
df_plot = df.query("method in ['riemannalignment', 'raw', 'spatiotemp']")
df_tab = df_plot[['subject_test', 'dataset_target', 'method', 'acc']]
df_tab = df_tab.groupby(['dataset_target', 'method',]).agg({"acc": ["mean", "std"]})
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.acc['mean']:.2f} $\pm$ {x.acc['std']:.2f}", axis=1 # noqa
)
# pivot for method and dataset_t
df_tab = df_tab.reset_index().pivot(index="method", columns="dataset_target", values="mean_std")
print(df_tab.to_latex(escape=False))

# %%

# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon
from matplotlib.pyplot import text

from sklearn.metrics import balanced_accuracy_score
# %%
fnames = list(Path("results/LODO_final/").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["bal"] = df.apply(
    lambda x: balanced_accuracy_score(x.y_target, x.y_pred),
    axis=1
)
df = pd.concat([df.query("method != 'spatiotemp'"), df.query("method == 'spatiotemp' & num_iter == 1")])
# # %%
# fnames = list(Path("results/valid_params/").glob("*.pkl"))
# df_spatio = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# df_spatio["bal"] = df_spatio.apply(
#     lambda x: balanced_accuracy_score(x.y_target, x.y_pred),
#     axis=1
# )
# # df_spatio = df_spatio.query("filter_size == 2048 & reg == 1e-7")
# df_spatio = df_spatio.query("reg == 0.001 & filter_size == 256")
# df_spatio["bal_adapted"] = df_spatio["bal"]

# %% Plot Spatio-temp vs RA and raw
pairs = [
    ("raw", "spatiotemp"),
    ("raw", "riemann"),
    ("spatiotemp", "riemann"),
]

# df_plot = pd.concat(
#     (df.query("method in ['raw', 'riemann']"), df_spatio), axis=0
# )
df_plot = df.query("method in ['raw', 'riemann', 'spatiotemp']")
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]
df_plot = df_plot.groupby(
    ["subject", "dataset_t", "method"]
).bal.mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True,)  # sharey=True)
for i, ax in enumerate(axes.flatten()):
    df_plot_ = df_plot.query(f"dataset_t == '{dataset[i]}'")

    raw = df_plot_.loc[(df_plot_.method == "raw"), "bal"].values
    riemann = df_plot_.loc[(df_plot_.method == "riemann"), "bal"].values
    spatiotemp = df_plot_.loc[(df_plot_.method == "spatiotemp"), "bal"].values

    pvalues = [
        wilcoxon(raw, spatiotemp, alternative="two-sided").pvalue,
        wilcoxon(raw, riemann, alternative="two-sided").pvalue,
        wilcoxon(spatiotemp, riemann, alternative="two-sided").pvalue
    ]

    axis = sns.boxplot(
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="bal",
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
        x="bal",
        y="method",
        orient="h",
        palette="colorblind",
        showfliers=False,
    )
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    sns.stripplot(
        data=df_plot_,
        x="bal",
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
    acc = df_plot_.groupby("method").bal.mean().reset_index().bal
    std = df_plot_.groupby("method").bal.std().reset_index().bal
    labels = [
        f"No Align. \n ({acc[0]:.2f} $\pm$ {std[0]:.2f}) ", # noqa
        f"RA \n ({acc[1]:.2f} $\pm$ {std[1]:.2f})", # noqa
        f"STMA  \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})", # noqa
        # f"Temp CMMN  \n ({acc[3]:.2f} $\pm$ {std[3]:.2f})"
    ]
    ax.set_yticklabels(labels)
    ax.set_title(f"Target: {dataset[i]}  ({int(len(df_plot_)/3)} subj.)")
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("results/plots/LODO.pdf", bbox_inches="tight")
    # plt.xlabel("BACC")
# %%

# df_plot = df.groupby(["subject", "dataset_t", "method"]).bal.mean().reset_index()
df_plot = df.query("method == 'raw'")[["subject", "dataset_t", "bal", "seed"]].merge(
    df.query("method == 'spatiotemp'")[["subject", "dataset_t", "bal", "seed"]],
    on=["subject", "dataset_t", "seed"],
    suffixes=("", "_adapted"),
)
df_plot["delta"] = df_plot.bal_adapted - df_plot.bal
fig, axes = plt.subplots(
    1, 4, figsize=(7, 2), sharex=True, sharey=True, layout="constrained"
)
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]

cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)

fig.tight_layout(rect=[0, 0, .9, 1])
vmax = df_plot['delta'].max()
vmin = df_plot['delta'].min()
for i, ax in enumerate(axes.flatten()):
    df_plot_ = df_plot.query(f"dataset_t == '{dataset[i]}'")

    sns.scatterplot(
        data=df_plot_,
        x="bal",
        y="bal_adapted",
        alpha=0.5,
        linewidth=0.1,
        # marker=".",
        # change line color
        # edgecolor="blue",
        s=15,
        ax=ax,
        c=df_plot_["delta"].values,
        cmap=cmap,
        # palette=cmap,
        legend=False,
        vmin=vmin,
        vmax=vmax,
    )
    n = np.sum(df_plot_["delta"] > 0)
    text(
        0.17, 0.95, f"{np.round(n/len(df_plot_)*100, 2)}%",
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        size=7
    )
    # text(0.23, 0.07, f"{int(np.round((1 - n/len(df_plot_))*100))}%", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=9)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_title(f"Target: {dataset[i]} \n({int(len(df_plot_)/10)} subj.)", fontsize=10)

    ax.set_xlabel("BACC with No Align.", fontsize=9)
    if i == 0:
        ax.set_ylabel("BACC with STMA", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.set_ylim(lims)
    # change ticks size
    ax.tick_params(axis='both', which='major', labelsize=8)
norm = plt.Normalize(vmin, vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([.89, .21, .014, .63])
fig.colorbar(sm, cax=cbar_ax, label="$\Delta$BACC")  # noqa
# plt.suptitle("BACC for LODO with 4 datasets")
# plt.tight_layout()
fig.subplots_adjust(wspace=0.1,)

fig.savefig("results/plots/scatter_sleep.pdf", bbox_inches="tight")
# %%
df_plot = df.groupby(["subject", "dataset_t", "method"]).bal.mean().reset_index()
df_plot = df_plot.query("method == 'raw'")[["subject", "dataset_t", "bal"]].merge(
    df_plot.query("method == 'spatiotemp'")[["subject", "dataset_t", "bal"]],
    on=["subject", "dataset_t"],
    suffixes=("", "_adapted"),
)
df_plot = df_plot[["subject", "dataset_t", "bal_adapted", "bal"]]
df_plot["delta"] = df_plot.bal_adapted - df_plot.bal
# %%
df_list = []
for dataset in ["ABC", "CHAT", "HOMEPAP", "MASS"]:
    n = len(df_plot.query(f"dataset_t == '{dataset}'"))
    df_ = df_plot.query(
        f"dataset_t == '{dataset}'"
    ).sort_values("bal", ascending=True).iloc[:int(n * 0.2)]
    df_list.append(df_)
df_20 = pd.concat(df_list, axis=0)
df_20 = df_20.groupby("dataset_t").agg({"delta": ["mean", "std"]})
df_20["mean_std"] = df_20.apply(
    lambda x: f"{x.delta['mean']:.2f} $\pm$ {x.delta['std']:.2f}", axis=1 # noqa
)
df_20 = df_20[["mean_std"]]
# %%
df_all = df_plot.groupby("dataset_t").agg({"delta": ["mean", "std"]})
df_all["mean_std"] = df_all.apply(
    lambda x: f"{x.delta['mean']:.2f} $\pm$ {x.delta['std']:.2f}", axis=1 # noqa
)
df_all = df_all[["mean_std"]]
df_tab = pd.concat((df_20, df_all), axis=1)
df_tab = df_tab.round(2)


print(df_tab.to_latex(escape=False))

# %%
# print ablation study
pairs = [
    ("spatiotemp", "temp"),
    ("spatiotemp", "spatio"),
]

df_plot = df.query("method in ['temp', 'raw', 'spatio', 'spatiotemp']")

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True,)  # sharey=True)
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]
df_plot = df_plot.groupby(
    ["subject", "dataset_t", "method"]
).bal.mean().reset_index()
palette_base = sns.color_palette("colorblind")
palette = sns.color_palette(
    [palette_base[0], palette_base[1], palette_base[3], palette_base[2]]
)

for i, ax in enumerate(axes.flatten()):
    df_plot_ = df_plot.query(f"dataset_t == '{dataset[i]}'")

    temp = df_plot_.loc[(df_plot_.method == "temp"), "bal"].values
    spatiotemp = df_plot_.loc[(df_plot_.method == "spatiotemp"), "bal"].values
    spatio = df_plot_.loc[(df_plot_.method == "spatio"), "bal"].values

    pvalues = [
        wilcoxon(spatiotemp, temp, alternative="two-sided").pvalue,
        wilcoxon(spatiotemp, spatio, alternative="two-sided").pvalue,
    ]

    order = ["raw", "temp", "spatio", "spatiotemp"]
    axis = sns.boxplot(
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="bal",
        y="method",
        orient="h",
        palette=palette,
        showfliers=False,
        ax=ax,
        order=order,
    )
    annotator = Annotator(
        ax,
        pairs,
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="bal",
        y="method",
        orient="h",
        palette=palette,
        showfliers=False,
        order=order,
    )
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    sns.stripplot(
        data=df_plot_,
        x="bal",
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
        order=order,
    )
    acc = df_plot_.groupby("method").bal.mean().reset_index().bal
    std = df_plot_.groupby("method").bal.std().reset_index().bal
    labels = [
        f"No Align. \n ({acc[0]:.2f} $\pm$ {std[0]:.2f}) ", # noqa
        f"TMA  \n ({acc[3]:.2f} $\pm$ {std[3]:.2f})",
        f"SMA\n ({acc[1]:.2f} $\pm$ {std[1]:.2f})", # noqa
        f"STMA \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})", # noqa
    ]
    ax.set_yticklabels(labels)
    ax.set_title(f"Target: {dataset[i]} ({int(len(df_plot_)/4)} subj.)")
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("results/plots/LODO_ablation.pdf", bbox_inches="tight")
    # plt.xlabel("BACC")
# %%
df_tab = df_plot[['subject', 'dataset_t', 'method', 'bal']]
df_tab = df_tab.groupby(['dataset_t', 'method',]).agg({"bal": ["mean", "std"]})
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.bal['mean']:.2f} $\pm$ {x.bal['std']:.2f}", axis=1 # noqa
)
# pivot for method and dataset_t
df_tab = df_tab.reset_index().pivot(index="method", columns="dataset_t", values="mean_std")
print(df_tab.to_latex(escape=False))
# %%
# print ablation study
pairs = [
    ("spatiotemp", "temp"),
    ("spatiotemp", "spatio"),
]

df_plot = pd.concat((df.query("method in ['raw', 'DECISION']"), df_spatio), axis=0)

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True,)  # sharey=True)
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]
df_plot = df_plot.groupby(
    ["subject", "dataset_t", "method"]
).bal.mean().reset_index()
palette_base = sns.color_palette("colorblind")
palette = sns.color_palette(
    [palette_base[0], palette_base[1], palette_base[3], palette_base[2]]
)

for i, ax in enumerate(axes.flatten()):
    df_plot_ = df_plot.query(f"dataset_t == '{dataset[i]}'")

    # temp = df_plot_.loc[(df_plot_.method == "temp"), "bal"].values
    # spatiotemp = df_plot_.loc[(df_plot_.method == "spatiotemp"), "bal"].values
    # spatio = df_plot_.loc[(df_plot_.method == "spatio"), "bal"].values

    # pvalues = [
    #     wilcoxon(spatiotemp, temp, alternative="two-sided").pvalue,
    #     wilcoxon(spatiotemp, spatio, alternative="two-sided").pvalue,
    # ]

    order = ["raw", "DECISION", "spatiotemp"]
    axis = sns.boxplot(
        data=df_plot_,
        fliersize=3,
        width=0.9,
        x="bal",
        y="method",
        orient="h",
        palette=palette,
        showfliers=False,
        ax=ax,
        order=order,
    )
    # annotator = Annotator(
    #     ax,
    #     pairs,
    #     data=df_plot_,
    #     fliersize=3,
    #     width=0.9,
    #     x="bal",
    #     y="method",
    #     orient="h",
    #     palette=palette,
    #     showfliers=False,
    #     order=order,
    # )
    # annotator.set_pvalues(pvalues)
    # annotator.annotate()
    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    sns.stripplot(
        data=df_plot_,
        x="bal",
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
        order=order,
    )
    acc = df_plot_.groupby("method").bal.mean().reset_index().bal
    std = df_plot_.groupby("method").bal.std().reset_index().bal
    labels = [
        f"No Align. \n ({acc[1]:.2f} $\pm$ {std[1]:.2f}) ", # noqa
        f"DECISION\n ({acc[0]:.2f} $\pm$ {std[0]:.2f})", # noqa
        f"STMA \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})", # noqa
    ]
    ax.set_yticklabels(labels)
    ax.set_title(f"Target: {dataset[i]} ({int(len(df_plot_)/4)} subj.)")
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    # fig.savefig("results/plots/LODO_ablation.pdf", bbox_inches="tight")
    # plt.xlabel("BACC")
# %%

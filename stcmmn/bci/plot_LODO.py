# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
fnames = list(Path("results/LODO").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_filtered = df.query(
    "filter == True"
    "& n_epochs == 200"
    "& archi == 'ShallowNet'"
    "& method != 'spatiotemp'"
)
# %%
fnames = list(Path("results/sensitivity_study/").glob("*.pkl"))
df_spatio = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_spatio_filtered = df_spatio.query(
    "filter == True"
    "& n_epochs == 200"
    "& method == 'spatiotemp'"
    "& archi == 'ShallowNet'"
    "& filter_size == 16"
    "& reg == 0"
)

# %%
df = pd.concat((df_filtered, df_spatio_filtered), axis=0)
# %%
fig, axes = plt.subplots(
    2, 2, figsize=(8, 5), sharex=True,  # sharey=True
)
dataset = ["BNCI2014001", "PhysionetMI", "Weibo2014"]   
df_plot = df.groupby(
    ["subject_test", "dataset_target", "method"]
).mean().reset_index()
for i, ax in enumerate(axes.flatten()):
    if i == 3:
        continue
    df_plot_ = df_plot.query(f"dataset_target == '{dataset[i]}'")

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

    sns.swarmplot(
        data=df_plot_,
        x="acc",
        y="method",
        orient="h",
        dodge=False,
        legend=False,
        ax=ax,
        alpha=0.6,
        size=4,
        color="grey",
        edgecolor="black",
        linewidth=0.5,
    )
    acc = df_plot_.groupby("method").mean().reset_index().acc
    std = df_plot_.groupby("method").std().reset_index().acc
    labels = [
        f"w/o Normalization \n ({acc[0]:.2f} $\pm$ {std[0]:.2f}) ",
        f"Pre-Whitening \n ({acc[1]:.2f} $\pm$ {std[1]:.2f})",
        f"Spatio-Temp CMMN  \n ({acc[2]:.2f} $\pm$ {std[2]:.2f})",
        f"Temp CMMN  \n ({acc[3]:.2f} $\pm$ {std[3]:.2f})"
    ]
    # ax.set_yticklabels(labels)
    ax.set_title(f"Target: {dataset[i]}")
    ax.set_ylabel("")
ax.set_xlabel("ACC")

plt.suptitle("BACC for LODO with 3 datasets")
fig.savefig("results/plots/LODO.pdf", bbox_inches="tight")
plt.tight_layout()
# %%
fnames = list(Path("results/LODO").glob("*.pkl"))
df_test = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_test = df_test.query(
    "filter == True"
    "& n_epochs == 200"
    "& archi == 'ShallowNet'"
)
# %%
sns.boxplot(
    data=df_test.query("dataset_target == 'Schirrmeister2017'"),
    fliersize=3,
    width=0.9,
    x="acc",
    y="method",
    orient="h",
    palette="colorblind",
    showfliers=False,
)
# %%
df_test.groupby(["dataset_target", "method"]).mean().reset_index()
# %%
# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import balanced_accuracy_score

# %% Plot Spatio-temp vs RA and raw
fnames = list(Path("results/LODO").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df = df.query(
    "filter == True"
    "& n_epochs == 200"
    "& archi == 'ShallowNet'"
)
# %%

fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
axes[1][2].set_visible(False)
dataset = ["BNCI2014001", "PhysionetMI", "Weibo2014", "Cho2017", "Schirrmeister2017"]   
df_plot = df.groupby(
    ["subject_test", "dataset_target", "method"]
).mean().reset_index()
df_plot = df_plot.query("method in ['raw', 'riemannalignment', 'spatiotemp']")
for i, ax in enumerate(axes.flatten()):
    if i == 5:
        continue
    df_plot_ = df_plot.query(f"dataset_target == '{dataset[i]}'")

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
df_plot = pd.concat(
    (df.query("method == 'raw'"), df.query("method == 'spatiotemp'").rename(columns={"acc": "acc_adapted"})),
    axis=1
)
df_plot = df_plot[["subject_test", "dataset_target", "acc_adapted", "acc"]]
df_plot["delta"] = df_plot.acc_adapted - df_plot.acc
fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=True, layout="constrained")
axes[1][2].set_visible(False)

cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)


fig.tight_layout(rect=[0, 0, .9, 1])
vmax = df_plot['delta'].max()
vmin = df_plot['delta'].min()
for i, ax in enumerate(axes.flatten()):
    if i == 5:
        continue
    df_plot_ = df_plot.query(f"dataset_target == '{dataset[i]}'")
    sns.scatterplot(
        data=df_plot_,
        x="acc",
        y="acc_adapted",
        alpha=0.7,
        # linewidth=1,
        ax=ax,
        c=df_plot_["delta"].values,
        cmap=cmap,
        # palette=cmap,
        legend=False,
        vmin=vmin,
        vmax=vmax,
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    # ax.set_xlim(lims)
    ax.set_title(f"Target: {dataset[i]}")

    if i >= 2:
        ax.set_xlabel("No adapt")
    else:
        ax.set_xlabel("")
    if i % 2 == 0:
        ax.set_ylabel("With CMMN")
    else:
        ax.set_ylabel("")
    # ax.set_ylim(lims)

norm = plt.Normalize(vmin, vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([.91, .3, .03, .4])
fig.colorbar(sm, cax=cbar_ax, label="$\Delta$BACC")  # noqa
# plt.suptitle("BACC for LODO with 4 datasets")
# plt.tight_layout()

fig.savefig("results/plots/scatter_bci.pdf", bbox_inches="tight")
# %%
df_plot = pd.concat(
    (df.query("method == 'raw'"), df_spatio[["acc_adapted"]]),
    axis=1
)
df_plot = df_plot[["subject", "dataset_t", "bal_adapted", "bal"]]
df_plot["delta"] = df_plot.bal_adapted - df_plot.bal
# %%
df_list = []
for dataset in ["ABC", "CHAT", "HOMEPAP", "MASS"]:
    n = len(df_plot.query(f"dataset_t == '{dataset}'"))
    df_ = df_plot.query(
        f"dataset_t == '{dataset}'"
    ).sort_values("bal", ascending=True).iloc[:int(n) * 0.2]
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
    ax.set_title(f"Target: {dataset[i]}")
    if i >= 2:
        ax.set_xlabel("BACC")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig("results/plots/LODO_ablation_bci.pdf", bbox_inches="tight")
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

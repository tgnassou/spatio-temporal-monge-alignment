# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# %%
archi = "ShallowNet"

# Load the results of parameter sensitivity
fnames = list(Path("results/sensitivity_study/").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_filtered = df.query(
    "filter == True"
    "& method == 'spatiotemp'"
    f"& archi == '{archi}'"
)
df_mean = df_filtered.groupby(
    ["seed", "reg", "filter_size", "dataset_target"]
).mean().reset_index()

# load the baseline results
fnames = list(Path("results/LODO/3datasets").glob("*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_baseline_filtered = df_baseline.query(
    "filter == True"
    "& n_epochs == 200"
    f"& archi == '{archi}'"
)
df_basline_mean = df_baseline_filtered.groupby(
    ["dataset_target", "method"]
).mean().reset_index()

# Load Riemann baseline
# fnames = list(Path("results/LODO/").glob("*Rie*.pkl"))
# df_riemann_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# df_riemann_baseline_mean = df_riemann_baseline.groupby(
#     ["dataset_target", "method"]
# ).mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
dataset = ["BNCI2014001", "Weibo2014", "PhysionetMI",]
for i, ax in enumerate(axes.flatten()):
    if i == 3:
        continue
    sns.lineplot(
        data=df_mean.query(f"dataset_target == '{dataset[i]}'"),
        x="filter_size",
        y="acc",
        hue="reg",
        palette="colorblind",
        ax=ax,
        marker="o",
        markersize=8,
        linewidth=2,
        alpha=0.8,
        errorbar=("sd", 0.1),
        legend=False if i != 1 else True,
    )
    # plot baseline horizontal line
    if i == 1:
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        ax.get_legend().set_title("Regularization")
    line1 = ax.axhline(
        df_basline_mean.query(f"dataset_target == '{dataset[i]}' & method == 'raw'")["acc"].values[0],
        linestyle="--",
        color="k",
        label="No Aligment",
        alpha=0.8,
    )
    line2 = ax.axhline(
        df_basline_mean.query(f"dataset_target == '{dataset[i]}' & method == 'riemannalignment'")["acc"].values[0],
        linestyle="-.",
        color="k",
        label="Pre-Whitening",
        alpha=0.8,
    )
    # if i == 2:
    #     first_legend = ax.legend(handles=[line1, line2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #     ax.get_legend().set_title("ShallowFBCSPNet")
    #     plt.gca().add_artist(first_legend)
    # line3 = ax.axhline(
    #     df_riemann_baseline_mean.query(f"dataset_target == '{dataset[i]}' & method == 'Riemann'")["acc"].values[0],
    #     linestyle="--",
    #     color="red",
    #     label="No TSupdate",
    #     alpha=0.8,
    # )
    # line4 = ax.axhline(
    #     df_riemann_baseline_mean.query(f"dataset_target == '{dataset[i]}' & method == 'CenteredRiemann'")["acc"].values[0],
    #     linestyle="-.",
    #     color="red",
    #     label="TSUpdate",
    #     alpha=0.8,
    # )
    # if i == 3:
    #     ax.legend(handles=[line3, line4], bbox_to_anchor=(1.05, 0.1), loc=3, borderaxespad=0.)
    #     ax.get_legend().set_title("Riemann")
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    ax.set_title("Dataset Target: " + dataset[i])
    ax.set_xlabel("Filter size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 0.8)
    
plt.suptitle("Parameter sensitivity for LODO on 3 BCI datasets")
fig.savefig("results/plots/valid_params_BCI.pdf", bbox_inches="tight")
# %%
# give max per dataset
df_mean_2 = df_mean.groupby(["dataset_target", "filter_size", "reg"]).mean().reset_index()
idx = df_mean_2.groupby(["dataset_target"])['acc'].idxmax()
# %%
print(df_mean_2.loc[idx][["dataset_target", "filter_size", "reg", "acc"]].to_latex())
# %%
# give max per dataset
df_mean_2 = df_mean.groupby(["filter_size", "reg"]).mean().reset_index()
idx = df_mean_2['acc'].idxmax()
# %%
print(df_mean_2.loc[idx][["filter_size", "reg", "acc"]].to_latex())
# %%
fnames = list(Path("results/LODO/").glob("*Rie*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# %%
df_basline_mean = df_baseline.groupby(
    ["dataset_target", "method"]
).mean().reset_index()
# %%

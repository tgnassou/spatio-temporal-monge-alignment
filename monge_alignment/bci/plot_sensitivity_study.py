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
).acc.mean().reset_index()

fnames = list(Path("results/LODO_test/").glob("*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_baseline_filtered = df_baseline.query(
    "filter == True"
    "& n_epochs == 200"
    f"& archi == '{archi}'"
)
df_basline_mean = df_baseline_filtered.groupby(
    ["dataset_target", "method"]
).acc.mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
dataset = ["BNCI2014001", "Weibo2014", "PhysionetMI", "Cho2017", "Schirrmeister2017"]
sns.lineplot(
    data=df_mean,
    x="filter_size",
    y="acc",
    hue="dataset_target",
    palette="colorblind",
    ax=ax,
    marker="o",
    markersize=8,
    linewidth=2,
    alpha=0.8,
    errorbar=("sd", 0.1),
    legend=True,
)
# plot baseline horizontal line
ax.legend(fontsize="10")#bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
ax.get_legend().set_title("Dataset")

ax.set_xscale("log")
ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
# ax.set_title("Dataset Target: " + dataset[i])
# increase font of ticks
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("Filter size", fontsize=16)
ax.set_ylabel("Accuracy", fontsize=16)
# ax.set_ylim(0.55, 0.8)
plt.suptitle("BCI datasets", fontsize=16)
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
fnames = list(Path("results/LODO_test/").glob("*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# %%

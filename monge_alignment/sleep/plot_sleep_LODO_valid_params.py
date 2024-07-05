# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import balanced_accuracy_score
plt.rcParams['legend.title_fontsize'] = 10
# %%
fnames = list(Path("results/valid_params/").glob("*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["bal"] = df.apply(lambda x: balanced_accuracy_score(x.y_target, x.y_pred), axis=1)

# %%
fnames = list(Path("results/LODO/").glob("*.pkl"))
df_baseline = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df_baseline["bal"] = df_baseline.apply(lambda x: balanced_accuracy_score(x.y_target, x.y_pred), axis=1)
df_baseline = df_baseline.query("method == 'raw'").groupby(["dataset_t"]).bal.mean().reset_index()
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True) #,sharey=True)
dataset = ["ABC", "CHAT", "HOMEPAP", "MASS"]
df_plot = df.groupby(["dataset_t", "filter_size", "reg"]).bal.mean().reset_index()
sns.lineplot(
    data=df_plot.query("filter_size <= 512"),
    x="filter_size",
    y="bal",
    hue="dataset_t",
    palette="colorblind",
    ax=ax,
    marker="o",
    markersize=8,
    linewidth=2,
    alpha=0.8,
    legend=True,
)
# plot baseline horizontal line
ax.legend(fontsize="10")#bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
ax.get_legend().set_title("Dataset")

ax.set_xscale("log")
ax.set_xticks([8, 16, 32, 64, 128, 256, 512,])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.set_xlabel("Filter size", fontsize=16)
ax.set_ylabel("Accuracy",  fontsize=16)
# ax.set_ylim(0.55, 0.8)
ax.tick_params(axis="both", which="major", labelsize=12)
plt.suptitle("Sleep staging datasets", fontsize=16)
fig.savefig("results/plots/valid_params.pdf", bbox_inches="tight",)
# %%

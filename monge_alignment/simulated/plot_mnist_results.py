# %%
import pandas as pd
import seaborn as sns
import spiderplot as sp
import matplotlib.pyplot as plt
# %%
df = pd.read_pickle("data/MNIST/results_one_sided_small_sigma.pkl")
df_mean = df.groupby(["method", "domain"]).mean().reset_index()
# %%
df_mean_sym = df_mean.copy()
# add 180 to dir
df_mean_sym["dir"] = df_mean_sym["dir"] + 180

df_plot = pd.concat([df_mean, df_mean_sym])

df_plot["dir"] = df_plot["dir"].apply(lambda x: int(x))
# change MA to Monge Alignment
df_plot["method"] = df_plot["method"].apply(lambda x: "Monge Align." if x == "MA" else x)
# %%
fig = plt.figure(figsize=(5, 5))
sns.set_style("whitegrid")
# Create spider plot.
ax = sp.spiderplot(x="dir", y="accuracy", hue="method", legend=True,
                   data=df_plot, palette="colorblind", rref=0,)
# ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
# Adjust limits in radial direction.
ax.set_rlim([0.7, 1])
ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.4, 1.),
    borderaxespad=0.
)
ax.set_xlabel("Direction (degrees)")
plt.show()
# fig.savefig("plots/spider_plot.pdf", bbox_inches="tight")
# %%
import seaborn as sns
sns.scatterplot(data=df, x="dir", y="accuracy", hue="method")

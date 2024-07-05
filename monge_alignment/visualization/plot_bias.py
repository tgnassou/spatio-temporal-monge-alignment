# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("crest", n_colors=5)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# %%


def bias_function(f, length, rho=0.7):
    bias = rho**f/(1 - rho)
    for i in range(f):
        bias += rho**i * i/length
    return bias


def correlation_bound_function(rho, k):
    return rho**np.abs(k)


def var_function(f, length, n_chans=1):
    # c = 5/(2*np.floor(length/f))*(np.log(5*f**2) + 32*np.log(4*10**(2*n_chans)))
    c = f*np.log(f)
    return np.max([c, np.sqrt(c)])
# %%
length = 3000
fs = np.linspace(1, length, 20, dtype=int)
rhos = [0.995, 0.996, 0.997, 0.998, 0.999]
bias = np.zeros((len(rhos), len(fs)))
for i, rho in enumerate(rhos):
    for j, f in enumerate(fs):
        bias[i, j] = bias_function(f, length, rho)
    bias[i] = bias[i]/np.max(bias[i])  # normalize

var = np.zeros((len(fs)))
for i, f in enumerate(fs):
    var[i] = var_function(f, length)
var = var/np.max(var)

# normalize var
# var = var/np.max(var)
# %%
fig = plt.figure(figsize=(5, 3))
for i, rho in enumerate(rhos):
    plt.plot(
        fs, bias[i], color=palette[i],
        linewidth=2, alpha=0.8, label=rf"$\rho$={rho}"
    )
for i, rho in enumerate(rhos):
    plt.plot(
        fs, bias[i]+var, color=palette[i], linestyle="-.",
        linewidth=2, alpha=0.8,
    )
plt.plot(fs, var, color="black", linestyle="--",)
plt.ylabel("Bias")
plt.xlabel("$f$")
plt.legend()
fig.savefig("plots/bias.svg", bbox_inches="tight")

# %%
correlation_bound = np.zeros((len(rhos), 2*length))
for i, rho in enumerate(rhos):
    for j in range(-length, length):
        correlation_bound[i, j+length] = correlation_bound_function(rho, j)

# %%
fig = plt.figure(figsize=(5, 3))
for i, rho in enumerate(rhos):
    plt.plot(
        list(range(-length, length)),
        correlation_bound[i],
        color=palette[i],
        linewidth=2,
        alpha=0.9,
        label=rf"$\rho$={rho}"
    )
plt.ylabel("Correlation bound")
plt.xlabel("Time lag")
plt.legend()
fig.savefig("plots/correlation_bound.pdf", bbox_inches="tight")
# %%

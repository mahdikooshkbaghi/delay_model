import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.style.use('seaborn-white')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('font', family='serif')
plt.rc('font', family='serif')


def allelic_manifold(alpha, c, Nx, Ny, w):
    num = w**2+(alpha+1)*w
    denum = num+alpha
    x = num/denum + Nx
    num = (c*w)**2+(c*w)*(alpha+1)
    denum = num+alpha
    y = num/denum + Ny
    return np.log(x), np.log(y)


parser = argparse.ArgumentParser(
    description="Postprocess the netcfd file")
parser.add_argument("-f", "--filename", type=str)
args = parser.parse_args()

res_az = az.from_netcdf(args.filename)
# Training Data
x = res_az['observed_data']['obs'][:, 0]
y = res_az['observed_data']['obs'][:, 1]

# Summary statistics
summary_df = az.summary(res_az, kind='stats')
summary_df.to_csv('summary_statistics.csv.gz',  compression='gzip')
print(summary_df)

w_infer = summary_df[summary_df.index.str.startswith('w')].copy()
w_infer.sort_values(by='mean', inplace=True, ascending=False)
fig, ax = plt.subplots(1, 1)
ss_idx = np.arange(len(w_infer))
ax.plot(ss_idx, w_infer['mean'].values)
ax.fill_between(ss_idx, w_infer['hdi_3%'], w_infer['hdi_97%'],
                color='C1', alpha=0.2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('site rank')
ax.set_ylabel('site strength')
plt.tight_layout()

# Plot chains and density for alpha and c
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
az.plot_trace(res_az, var_names=['alpha', 'c'],
              compact=True, combined=True, axes=axs)
plt.tight_layout()
plt.savefig('trace_chains.pdf')

# Manifold plot
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(x, y, alpha=0.1, rasterized=True)
alpha_mean = np.exp(summary_df['mean']['log_alpha'])
c_mean = np.exp(summary_df['mean']['log_c'])
Nx_mean = np.exp(summary_df['mean']['log_nx'])
Ny_mean = np.exp(summary_df['mean']['log_ny'])
w_mean = np.sort(w_infer['mean'])
x_mean, y_mean = allelic_manifold(alpha_mean, c_mean, Nx_mean, Ny_mean, w_mean)
ax.plot(x_mean, y_mean, c='C1', linewidth=4, label=r'Allelic manifold')
ax.set_xlabel(r'$\log \Psi_x$')
ax.set_ylabel(r'$\log \Psi_y$')
ax.legend()
plt.tight_layout()
plt.savefig('allelic_manifold.pdf')

# Posteriors for parameters
posteriors = res_az.posterior.stack(draws=("chain", "draw"))
fig, axs = plt.subplots(2, 2, figsize=(8, 8/1.6))
sns.kdeplot(posteriors['alpha'], ax=axs[0, 0], shade=True, color='C0')
sns.kdeplot(posteriors['c'], ax=axs[0, 1], shade=True, color='C1')
sns.kdeplot(posteriors['Nx'], ax=axs[1, 0], shade=True, color='C2')
sns.kdeplot(posteriors['Ny'], ax=axs[1, 1], shade=True, color='C3')

for i in range(2):
    for j in range(2):
        axs[i, j].set_ylabel('')
        axs[i, j].set_yticks([])

plt.tight_layout()
plt.savefig('parameters_kdeplots.pdf')

# Posterior Predictive plot
# Stack all the chains
obs_post_pred = res_az['posterior_predictive']['obs'].stack(
    draws=("chain", "draw"))
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
sns.kdeplot(x, ax=axs[0], label='observation', linewidth=2)
sns.kdeplot(y, ax=axs[1], label='observation', linewidth=2)
for i in range(100):
    if i == 0:
        label = 'Posterior Predictive'
    else:
        label = None
    sns.kdeplot(obs_post_pred[:, 0, i],
                ax=axs[0], c='C1', linewidth=0.1,
                label=label)

    sns.kdeplot(obs_post_pred[:, 1, i],
                ax=axs[1], c='C1', linewidth=0.1,
                label=label)
axs[0].set_xlabel(r'$\Psi_x$')
axs[1].set_xlabel(r'$\Psi_y$')
plt.tight_layout()
plt.savefig('Posterior_Predictive.pdf')

plt.show()

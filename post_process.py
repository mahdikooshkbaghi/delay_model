import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

res_az = az.from_netcdf('res_N50_C2_W20.nc')


# Training Data
x = res_az['observed_data']['obs'][:, 0]
y = res_az['observed_data']['obs'][:, 1]


# Plot original training data
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(x, y, alpha=0.1)
ax.set_xlabel(r'$\Psi$ BRCA2')
ax.set_ylabel(r'$\Psi$ IKBKAP')
plt.tight_layout()

# Summary statistics
summary_df = az.summary(res_az, kind='stats')
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
ax.set_xlabel('Splicing rank')
ax.set_ylabel('Splicing strength')
plt.tight_layout()

# Plot chains and density for alpha and c
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
az.plot_trace(res_az, var_names=['alpha', 'c'],
              compact=True, combined=True, axes=axs)
plt.tight_layout()

# Posterior Predictive plot
posterior_predictive = res_az['posterior_predictive']
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
sns.kdeplot(x, ax=axs[0], label='observation', linewidth=2)
sns.kdeplot(y, ax=axs[1], label='observation', linewidth=2)
for i in range(10):
    if i == 0:
        label = 'Posterior Predictive'
    else:
        label = None
    pp_obs = posterior_predictive['obs']
    for k in range(pp_obs.shape[0]):
        sns.kdeplot(pp_obs[k, i, :, 0],
                    ax=axs[0], c='C1', linewidth=0.1,
                    label=label)

    for k in range(pp_obs.shape[0]):
        sns.kdeplot(pp_obs[k, i, :, 1],
                    ax=axs[1], c='C1', linewidth=0.1,
                    label=label)

plt.show()

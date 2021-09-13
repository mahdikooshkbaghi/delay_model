import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import optimize, stats
import argparse


def get_mpsa_brca2_ikbkap_data():
    """
    1. Get data from the MPSA database (BRCA2/IKBKAP 2018).
    2. Devide Psi = PSI/100
    3. Returns the log(Psi)
    """
    psi_df = pd.read_csv('psi_5ss_2018_brca2_ikbkap_smn1.gz')
    psi_df.dropna(inplace=True)
    cond = (psi_df['brca2_9nt'] > 0) & (psi_df['ikbkap_9nt']) > 0
    psi_df = psi_df[cond]
    psi_df.reset_index(inplace=True, drop=True)
    brca2_psi = psi_df['brca2_9nt'].values/100.
    ikbkap_psi = psi_df['ikbkap_9nt'].values/100.

    return np.log(brca2_psi), np.log(ikbkap_psi)


def backg_noise_model(x_psi, y_psi):

    x_kernel = stats.gaussian_kde(x_psi)
    opt = optimize.minimize_scalar(lambda x: -x_kernel(x))
    Nx = float(opt.x[0])

    y_kernel = stats.gaussian_kde(y_psi)
    opt = optimize.minimize_scalar(lambda x: -y_kernel(x))
    Ny = float(opt.x[0])
    return Nx, Ny


def allelic_manifold(alpha, c, Nx, Ny, w):
    num = w**2+(alpha+1)*w
    denum = num+alpha
    x = num/denum + Nx
    num = (c*w)**2+(c*w)*(alpha+1)
    denum = num+alpha
    y = num/denum + Ny
    return np.log(x), np.log(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian inference for one step delay model")
    parser.add_argument("-n",
                        "--num_samples",
                        nargs="?",
                        default=200,
                        type=int)
    parser.add_argument("-w", "--num_warmup", nargs="?",
                        default=1000, type=int)
    parser.add_argument("-c", "--num_chains", nargs="?",
                        default=4, type=int)
    args = parser.parse_args()

    random_seed = 1234

    # Get data
    x, y = get_mpsa_brca2_ikbkap_data()
    # Find the fixed background noise values
    Nx, Ny = backg_noise_model(np.exp(x), np.exp(y))

    # Define pymc3 model
    with pm.Model() as model:
        log_alpha = pm.Uniform('log_alpha', lower=-10, upper=0)
        log_c = pm.Uniform('log_c', lower=-10, upper=1)
        log_w_mean = pm.Normal('log_w_mean', mu=0, sigma=1)
        log_w_sigma = pm.Gamma('log_w_sigma', mu=1, sigma=1)
        log_w_raw = pm.Normal('log_w_raw', mu=0, shape=x.shape[0])
        alpha = pm.Deterministic('alpha', pm.math.exp(log_alpha))
        c = pm.Deterministic('c', pm.math.exp(log_c))
        w = pm.Deterministic('w', pm.math.exp(
            log_w_mean + log_w_sigma * log_w_raw))
        num = w**2+(alpha+1)*w
        denum = num+alpha
        mu_x = pm.math.log(num/denum + Nx)
        num = (c*w)**2+(c*w)*(alpha+1)
        denum = num+alpha
        mu_y = pm.math.log(num/denum + Ny)
        x_likelihoods = pm.Normal('x_like', mu=mu_x, sigma=1, observed=x)
        y_likelihoods = pm.Normal('y_like', mu=mu_y, sigma=1, observed=y)

    # Check model
    print(model.check_test_point())

    # Prior predictive check
    with model:
        priors_checks = pm.sample_prior_predictive(
            samples=100, random_seed=random_seed)

    # Run the NUTS MCMC
    with model:
        step = pm.NUTS(target_accept=0.99)
        trace = pm.sample(draws=args.num_samples, step=step,
                          cores=args.num_chains,
                          tune=args.num_warmup, init='auto',
                          return_inferencedata=False,
                          random_seed=random_seed)

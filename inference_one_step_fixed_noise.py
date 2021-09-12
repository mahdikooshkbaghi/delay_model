import numpy as np
import argparse
from scipy import optimize
from scipy import stats
from numpy.random import default_rng
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import arviz as az
import pandas as pd
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
from numpyro.diagnostics import summary
from numpyro.infer import Predictive


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

    return jnp.log(brca2_psi), jnp.log(ikbkap_psi)


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


def model(len_ss, Nx, Ny, obs=None):
    log_alpha = numpyro.sample('log_alpha', dist.Uniform(-4, 4))
    log_c = numpyro.sample('log_c',  dist.Uniform(-8, 0))
    log_w_mean = numpyro.sample('log_w_mean', dist.Uniform(-1, 1))
    log_w_sigma = numpyro.sample('log_w_sigma', dist.Gamma(concentration=1,
                                                           rate=1))
    log_w_raw = numpyro.sample(
        'log_w_raw', dist.Normal(loc=jnp.zeros((len_ss,))))
    alpha = numpyro.deterministic('alpha', jnp.exp(log_alpha))

    c = numpyro.deterministic('c', jnp.exp(log_c))
    w = numpyro.deterministic('w', jnp.exp(
        log_w_mean + log_w_sigma * log_w_raw))

    num = w**2+(alpha+1)*w
    denum = num+alpha
    mu_x = jnp.log(num/denum + Nx)

    num = (c*w)**2+(c*w)*(alpha+1)
    denum = num+alpha
    mu_y = jnp.log(num/denum + Ny)

    sigma = numpyro.sample('sigma', dist.Normal(loc=0, scale=0.5))
    numpyro.sample('obs',
                   dist.Normal(loc=jnp.stack([mu_x, mu_y], -1), scale=sigma),
                   obs=obs)


def main(args):
    rng_jax = random.PRNGKey(0)
    rng_numpy = default_rng(1234)

    # Get data
    x_train, y_train = get_mpsa_brca2_ikbkap_data()
    # Filter data
    x = x_train
    y = y_train
    Nx, Ny = backg_noise_model(jnp.exp(x), jnp.exp(y))

    kernel = NUTS(model, target_accept_prob=0.99)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains)
    mcmc.run(rng_jax,
             len_ss=x.shape[0],
             Nx=Nx,
             Ny=Ny,
             obs=jnp.c_[x, y])
    # Convert the inference data to arviz for plotting and load later.
    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(random.PRNGKey(1),
                                                                Nx=Nx,
                                                                Ny=Ny,
                                                                len_ss=x.shape[0])
    prior = Predictive(model, num_samples=500)(
        random.PRNGKey(10), Nx=Nx, Ny=Ny, len_ss=x.shape[0])
    az_data = az.from_numpyro(
        mcmc, prior=prior, posterior_predictive=posterior_predictive)
    # Saving inference data to the netcfd format
    az_data.to_netcdf(
        f'res_N{args.num_samples}_C{args.num_chains}_W{args.num_warmup}_fixed_noise.nc')


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
    parser.add_argument("--device",
                        default="cpu",
                        type=str,
                        help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)

from jax import numpy as jnp


opt_beta_gamma = [[-0.11657826, -0.24156693, -0.27128321, -0.34703595],
                  [0.36039333, 0.23176248, 0.23125999, 0.1549918]]

fixed_opt_gamma = jnp.array(opt_beta_gamma[0][:-1])
fixed_opt_beta = jnp.array(opt_beta_gamma[1][:-1])

variational_opt_gamma = opt_beta_gamma[0][-1]
variational_opt_beta = opt_beta_gamma[1][-1]

opt_params = jnp.array([fixed_opt_gamma, fixed_opt_beta]).T

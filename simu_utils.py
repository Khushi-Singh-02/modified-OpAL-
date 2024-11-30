from learning_dash import simulate  # Ensure you have this import
import numpy as np 

def gridsimu(params, dynaPRL, save_warmup):
    param_combo = params

    alpha_a_Go = np.round(param_combo[0], 2)
    alpha_a_NoGo = np.round(param_combo[1], 2)
    alpha_c = param_combo[2]
    beta_GO = param_combo[3]
    beta_NOGO = param_combo[4]

    simulation_params = (alpha_c, alpha_a_Go, alpha_a_NoGo, beta_GO, beta_NOGO)

    data = simulate(
        simulation_params, n_states=1, n_trials=dynaPRL.n_trials, task=dynaPRL, v0=0.0, crit="SA",
        mod="avg_value", k=1, phi=1, rho=0, baselinerho=0.0, threshRho=0, threshRand=0,
        rnd_seeds=None, norm=False, mag=1, norm_type=None, hebb=True, variant=None,
        anneal=False, T=100.0, use_var=False, decay_to_prior=True, decay_to_prior_gamma=1.,
        save_warmup=save_warmup, disp=False
    )
    
    return data

# Function to wrap the simulation call
def gridsimu_wrapper(args):
    params, dynaPRL, save_warmup = args
    return gridsimu(params, dynaPRL, save_warmup)

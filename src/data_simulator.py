import numpy as np


def simulate_data(T=100, exp_omega=0.05, decay_factor=0.05, seed=1, type="constant"):
    """Simulate data with time varying average growth rate omega.

    Parameters
    ----------
    T : int
        Number of time periods.
    exp_omega : float
        Scaling factor of omega.
    decay_factor : float
        Speed of growth rae decay for decay model.
    type : str
        Type of time variation in omega, one of "constant", "sine", and "decay".

    Returns
    -------
    x, y, x_t, lam_t, omega
        Data and parameters.
    """

    np.random.seed(seed)

    if type == "decay":
        omega = np.log(exp_omega) + decay_factor * np.linspace(start=0, stop=-T, num=T)
    elif type == "sine":
        omega = (np.sin(np.linspace(start=0.01, stop=1, num=T)) + 1.5) * np.log(
            exp_omega
        )
    else:
        omega = np.repeat(np.log(exp_omega), T)

    x_t = [1]
    lam_t = []
    for t in range(T):
        lam_tp1 = np.sum(x_t) * np.exp(omega[t])
        x_tp1 = np.random.poisson(lam=lam_tp1)
        x_t.append(x_tp1)
        lam_t.append(lam_tp1)

    y = np.array(x_t)[1:]
    x = np.cumsum(x_t)[0:-1]
    x_t = np.array(x_t)[0:-1]

    return x, y, x_t, lam_t, omega

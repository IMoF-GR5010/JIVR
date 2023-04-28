import numpy as np

def forward_price(S, r_t_tau, q_t, tau):
    return np.multiply(S, np.exp(np.multiply(r_t_tau - q_t, tau)))

def moneyness(tau, F_t_tau, K):
    return np.multiply(np.power(tau, -0.5), np.log(np.divide(F_t_tau, K)))

def time_to_maturity(tau, T_conv):
    return np.exp(-np.power(np.divide(tau, T_conv), 1/2))

def moneyness_slope(M):
    return np.where(M >= 0, M, np.divide(np.exp(2*M) - 1, np.exp(2*M) + 1))

def smile_attenuation(M, tau, T_max):
    return np.multiply(1 - np.exp(-np.power(M, 2)), np.log(np.divide(tau, T_max)))

def smirk(M, tau, T_max):
    return np.where(M < 0, np.multiply(1 - np.exp(np.power(3*M, 3)), np.log(np.divide(tau, T_max))), 0)
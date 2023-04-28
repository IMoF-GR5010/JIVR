from .functions import forward_price, moneyness, time_to_maturity, moneyness_slope, smile_attenuation, smirk
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

class Implied_Volatility(object):

    def __init__(self, filepath=None, beta=None, T_conv=0.25, T_max=5):
        if filepath:
            self.beta = pd.read_csv(filepath)
        else:
            self.beta = beta
        self.T_conv = T_conv
        self.T_max = T_max

    def predict(self, S, r, q, tau, K, t=None):
        if t is None:
            t = S.index
            S_t, r_t, q_t, tau_t, K_t = S.to_numpy(), r.to_numpy(), q.to_numpy(), tau.to_numpy(), K.to_numpy()
        else:
            S_t, r_t, q_t, tau_t, K_t = S[t].to_numpy(), r[t].to_numpy(), q[t].to_numpy(), tau[t].to_numpy(), K[t].to_numpy()
        F_t_tau = forward_price(S_t, r_t, q_t, tau_t)
        M_t = moneyness(tau_t, F_t_tau, K_t)
        sigma_t = self.beta[:, 0] + \
                np.multiply(self.beta[:, 1], time_to_maturity(tau_t, self.T_conv)) + \
                np.multiply(self.beta[:, 2], moneyness_slope(M_t)) + \
                np.multiply(self.beta[:, 3], smile_attenuation(M_t, tau_t, self.T_max)) + \
                np.multiply(self.beta[:, 4], smirk(M_t, tau_t, self.T_max))
        sigma = pd.Series(data={'sigma':sigma_t}, index=t)
        return sigma
    
    def fit(self, sigma, S, r, q, tau, K):
        beta_t = np.empty((sigma.index.nunique(), 5))
        unique_time = sigma.index.sort_values().unique()
        for i, t in enumerate(unique_time):
            beta_t[i, :] = np.squeeze(self._fit_for_t(sigma[t], S[t], r[t], q[t], tau[t], K[t]))
        self.beta = pd.DataFrame(data={'beta_1':beta_t[:, 0], 'beta_2':beta_t[:, 1], 'beta_3':beta_t[:, 2], 'beta_4':beta_t[:, 3], 'beta_5':beta_t[:, 4]}, index=unique_time)
        self._adjust_params()

    def _fit_for_t(self, sigma, S, r, q, tau, K):
        n = sigma.size
        sigma_t = sigma.to_numpy().reshape((n, 1))
        S_t = S.to_numpy().reshape((n, 1))
        r_t = r.to_numpy().reshape((n, 1))
        q_t = q.to_numpy().reshape((n, 1))
        tau_t = tau.to_numpy().reshape((n, 1))
        K_t = K.to_numpy().reshape((n, 1))
        F_t_tau = forward_price(S_t, r_t, q_t, tau_t)
        M_t = moneyness(tau_t, F_t_tau, K_t)
        
        Y = sigma_t
        X = np.concatenate([np.ones((n, 1)), \
                            time_to_maturity(tau_t, self.T_conv), \
                            moneyness_slope(M_t), \
                            smile_attenuation(M_t, tau_t, self.T_max), \
                            smirk(M_t, tau_t, self.T_max)], axis=1)
        beta_t = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        return beta_t
    
    def _adjust_params(self, std_4_target=0.02, std_5_target=0.02):
        std_4 = np.std(self.beta.iloc[:, 3])/std_4_target
        self.beta.iloc[:, 3] = self.beta.iloc[:, 3]/std_4
        std_5 = np.std(self.beta.iloc[:, 4])/std_5_target
        self.beta.iloc[:, 4] = self.beta.iloc[:, 4]/std_5

    def save(self, filepath):
        self.beta.to_csv(filepath, index=True)
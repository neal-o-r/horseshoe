# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
from scipy.stats import bernoulli
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

plt.style.use('ggplot')

np.random.seed(123)

def make_data(n, m):

    alpha = 3
    sigma = 1
    sig_p = 0.05

    beta = np.zeros(m)
    f = np.zeros(m)
    for i in range(m):
        if bernoulli(sig_p).rvs():
            if bernoulli(0.5).rvs():
                beta[i] = np.random.normal(10, 1)
            else:
                beta[i] = np.random.normal(-10, 1)
            f[i] = 1

        else:
            beta[i] = np.random.normal(0, 0.25)

    X = np.random.normal(0, 1, (n, m))
    y = np.random.normal(X.dot(beta) + alpha, sigma)

    return X, y, beta, f


def plot_beta(b, b_hat, std=None):
    x = range(len(b))
    plt.plot(x, b, 'k', alpha=0.5)
    plt.plot(x, b_hat, alpha=0.5)

    if std is not None:
        plt.fill_between(x, b_hat + std, b_hat - std, alpha=0.3)

    plt.show()


def sk_lasso_model(X, y, b):
    ls = Lasso(fit_intercept=False)
    ls.fit(X, y)
    plot_beta(b, ls.coef_)


def pm_lasso_model(X, y, b):

    lasso = pm.Model()
    with lasso:
        beta = pm.Laplace('beta', 0, b=1, shape=X.shape[1])
        y_hat = tt.dot(X, beta)
        likelihood = pm.Normal('likelihood', y_hat, observed=y)

        trace = pm.sample(1000)

    b_hat = trace.get_values('beta').mean(0)
    b_sig = trace.get_values('beta').std(0)
    plot_beta(b, b_hat, std=b_sig)


    return trace


def pm_horseshoe(X, y, b):

    m = 10
    ss = 3
    dof = 25

    horseshoe = pm.Model()
    with horseshoe:
        sigma = pm.HalfNormal('sigma', 2)
        tau_0 = m / (X.shape[0] - m) * sigma / tt.sqrt(X.shape[0])

        tau = pm.HalfCauchy('tau', tau_0)
        c2  = pm.InverseGamma('c2', dof/2, dof/2 * ss**2)
        lam = pm.HalfCauchy('lam', 1)

        l1 = lam * tt.sqrt(c2)
        l2 = tt.sqrt(c2 + tau * tau * lam * lam)
        lam_d = l1 / l2

        beta = pm.Normal('beta', 0, tau * lam_d, shape=X.shape[1])
        y_hat = tt.dot(X, beta)

        likelihood = pm.Normal('likelihood', y_hat, observed=y)
        trace = pm.sample(1000)

    b_hat = trace.get_values('beta').mean(0)
    b_sig = trace.get_values('beta').std(0)
    plot_beta(b, b_hat, std=b_sig)


if __name__ == '__main__':

    X, y, b, f = make_data(100, 200)
    sk_lasso_model(X, y, b)




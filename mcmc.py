import numpy as np
import scipy.stats as stats

class MetropolisHastings(object):

    def __init__(self, n_samples=100, prop_init=0, prop_width=1, prior_mu=0, prior_sd=1):
        self.data = None
        self.n_samples = n_samples
        self.prop_init = prop_init
        self.prop_width = prop_width
        self.prior_mu = prior_mu
        self.prior_sd = prior_sd
        self.posterior = None
        self.acceptance_rate = None

    def fit(self, data):
        n_accepted = 0
        prop_current = self.prop_init
        posterior = [prop_current]
        for _ in range(self.n_samples):

            if (_+1) % 100 == 0:
                print(f'{(_+1)/self.n_samples * 100} %')

            prop_proposal = stats.norm(prop_current, self.prop_width).rvs()

            p_current = stats.norm(prop_current, 1).pdf(data).sum() * stats.norm(self.prior_mu, self.prior_sd).pdf(prop_current)
            p_proposal = stats.norm(prop_proposal, 1).pdf(data).sum() * stats.norm(self.prior_mu, self.prior_sd).pdf(prop_proposal)

            if np.random.rand() < np.min([1, p_proposal / p_current]):
                prop_current = prop_proposal
                n_accepted += 1
            
            posterior.append(prop_current)
        
        self.acceptance_rate = n_accepted / self.n_samples
        self.posterior = np.array(posterior)

class Gibbs(object):
    
    def __init__(self, mu, sigma, n_samples):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.n_samples = n_samples
        self.posterior = None

    def cond_xy(self, y):
        mu = self.mu[0] + self.sigma[1,0] / self.sigma[0,0] * (y - self.mu[1])
        sigma = self.sigma[0,0] - self.sigma[1,0] / self.sigma[1,1] * self.sigma[1,0]
        return np.random.normal(mu, sigma)

    def cond_yx(self, x):
        mu = self.mu[1] + self.sigma[0,1] / self.sigma[1,1] * (x - self.mu[0])
        sigma = self.sigma[1,1] - self.sigma[0,1] / self.sigma[0,0] * self.sigma[0,1]
        return np.random.normal(mu, sigma)

    def fit(self):
        posterior = {'x':[], 'y':[]}
        y = self.mu[1]

        for _ in range(self.n_samples):

            if (_+1) % 100 == 0:
                print(f'{(_+1)/self.n_samples * 100} %')

            x = self.cond_xy(y)
            y = self.cond_yx(x)
            posterior['x'].append(x)
            posterior['y'].append(y)
        self.posterior = posterior

class Hamiltonian(object):
    def __init__(self):
        pass

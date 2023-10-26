import numpy as np
from parameter_setting import *
from utils import *
from scipy.stats import wishart

class Agent:
    def __init__(self, dataset_name, agent_name, model_dir, observation_train_loader, observation_all_loader):
        self.dataset_name = dataset_name
        self.agent_name = agent_name
        self.model_dir = model_dir
        self.train_loader = observation_train_loader
        self.all_loader = observation_all_loader
        self.D = len(self.all_loader.dataset)
        self.w = np.random.choice(K, self.D)
        self.lam = wishart.rvs(df=nu, scale=beta, size=K)
        self.mu = np.array([np.random.multivariate_normal(mean=m, cov=np.linalg.inv(alpha * self.lam[k])).flatten() for k in range(K)])
        self.z = np.random.normal(0, 1, (self.D, latent_dim))
        self.prior_mu = np.zeros((self.D, latent_dim))
        self.prior_var = np.ones((self.D, latent_dim))

    def initialize_parameters(self):
        self.w = np.random.choice(K, self.D)
        self.lam = np.array([np.eye((latent_dim)) for _ in range(K)])
        self.mu = np.zeros((K, latent_dim))

    def propose(self):
        return propose_sampling(self.z, self.mu, self.lam)

    def accept_or_reject(self, proposed_w):
        count_accept = 0
        for d in range(self.D):
            r = acceptance_rate(proposed_w[d], self.w[d], self.z[d], self.mu, self.lam)
            u = np.random.rand()
            if u < r:
                self.w[d] = proposed_w[d]
                count_accept += 1

    def update(self):
        self.mu, self.lam = sampling_mu_lambda(self.w, self.z)
    
    def set_vae_prior(self):
        for d in range(self.D):
            self.prior_mu[d] = self.mu[self.w[d]]
            self.prior_var[d] = np.diag(np.linalg.inv(self.lam[self.w[d]]))

    def vae(self, iteration):
        self.z = vae_train(self.dataset_name, self.agent_name, self.model_dir, iteration ,self.train_loader, self.all_loader, self.prior_mu, self.prior_var)
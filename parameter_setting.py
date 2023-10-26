import numpy as np
import torch

# Hyperparameters for training
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Hyperparameters setting
K = 10
latent_dim = 12
lr = 1e-4
alpha = 0.01
nu = latent_dim
m = np.zeros(latent_dim)
beta = np.eye(latent_dim) * 0.05

# Hyperparameters for Mutual Inference
mi_iter = 10

# Hyperparameters for M-H
mh_iter = 100

# Hyperparameters for VAE
vae_iter = 100
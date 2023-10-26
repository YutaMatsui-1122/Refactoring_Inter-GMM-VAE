from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from tool import visualize_ls, get_param

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_dim = 12
h_dim = 1024
#h_dim = 2304
image_channels = 3
image_size = 64
data_rep = 4

img_dataSize = torch.Size([3, 32, 32])
img_imgChans = img_dataSize[0]
img_fBase = 32
const = 1e-6

class Flatten(nn.Module):
    def forward(self, input):
        #print("input.size(0)",input.size(0))
        #print("input.view(input.size(0),-1)",input.view(input.size(0), -1).size())
        return input.view(input.size(0), -1) # [10,1024]
class UnFlatten(nn.Module):
    def forward(self, input, size=h_dim):
        #print("input.view(input.size(0),size,1,1)",input.view(input.size(0), size,1,1).size())
        return input.view(input.size(0), size, 1, 1)
    
class VAE(nn.Module):
    def __init__(self, image_channels=image_channels, h_dim=h_dim, x_dim=x_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=6, stride=2),            
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        # 事前分布のパラメータN(0,I)で初期化
        self.prior_var = nn.Parameter(torch.Tensor(1, x_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, o_d):
        #print("o_d: ",o_d.size()) # [batch_size, channel, 縦, 横]
        h = self.encoder(o_d)
        #print("h: ",h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, o_d):
        z, mu, logvar = self.encode(o_d)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_o, o_d, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        # print(f"o_d : {o_d.size()}  recon_o:{recon_o.size()}")
        # print(f"recon_o:{recon_o.view(o_d.size()).size()}")
        BCE = F.binary_cross_entropy(recon_o, o_d, reduction='sum')
        beta = 1.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        if iteration != 0:
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.float().expand_as(en_mu).to(device)
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.float().expand_as(en_logvar).to(device)
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device)

            var_division = en_logvar.exp() / prior_var  # Σ_0 / Σ_1
            diff = en_mu - prior_mu  # μ_１ - μ_0
            diff_term = diff * diff / prior_var  # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
            logvar_division = prior_logvar - en_logvar  # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - x_dim)
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD



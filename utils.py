import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wishart, multivariate_normal
from parameter_setting import *
import cnn_vae_module_mnist
import cnn_vae_module_fruits
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset


class Dataset_mnist_index(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
class Dataset_fruits_index(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None: sample = self.transform(sample)
        if self.target_transform is not None: target = self.target_transform(target)
        return sample, target, index

def create_dataloader(dataset_name, batch_size, category):
    if dataset_name == "mnist":
        # MNIST
        angle_a = 0 # MNIST's angle for Agent A
        angle_b = 45 # MNIST's angle for Agent B
        trans_angA = transforms.Compose([transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.ToTensor()])
        trans_angB = transforms.Compose([transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.ToTensor()]) 
        # Define datasets
        trainval_datasetA = Dataset_mnist_index('./data/mnist', train=True, transform=trans_angA, download=True) # Dataset for Agent A
        trainval_datasetB = Dataset_mnist_index('./data/mnist', train=True, transform=trans_angB, download=True) # Dataset for Agent B
        n_samples = len(trainval_datasetA)
        D = int(n_samples * (1/6)) # Total data
        subset_indices = list(range(0, D))
        train_datasetA = Subset(trainval_datasetA, subset_indices)
        train_datasetB = Subset(trainval_datasetB, subset_indices)
        train_loaderA = torch.utils.data.DataLoader(train_datasetA, batch_size=batch_size, shuffle=True) # train_loader for agent A
        train_loaderB = torch.utils.data.DataLoader(train_datasetB, batch_size=batch_size, shuffle=True) # train_loader for agent B
        all_loaderA = torch.utils.data.DataLoader(train_datasetA, batch_size=batch_size, shuffle=False)
        all_loaderB = torch.utils.data.DataLoader(train_datasetB, batch_size=batch_size, shuffle=False)
    elif dataset_name == "fruits360":
        angle_a = 0 # Fruits360's angle for Agent A
        angle_b = 25 # Fruits360's angle for Agent B
        trans_angA = transforms.Compose([transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.Resize((64, 64)), transforms.ToTensor()])
        trans_angB = transforms.Compose([transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.Resize((64, 64)), transforms.ToTensor()]) 
        # Define datasets
        train_datasetA = Dataset_fruits_index(f"./data/Fruits360/{category}/VAE", transform= trans_angA) # Dataset for Agent A
        train_datasetB = Dataset_fruits_index(f"./data/Fruits360/{category}/VAE", transform= trans_angB) # Dataset for Agent B
        all_datasetA = Dataset_fruits_index(f"./data/Fruits360/{category}/MH", transform= trans_angA) # Dataset for Agent A
        all_datasetB = Dataset_fruits_index(f"./data/Fruits360/{category}/MH", transform= trans_angB) # Dataset for Agent B
        train_loaderA = torch.utils.data.DataLoader(train_datasetA, batch_size=batch_size, shuffle=True) # train_loader for agent A
        train_loaderB = torch.utils.data.DataLoader(train_datasetB, batch_size=batch_size, shuffle=True) # train_loader for agent B
        all_loaderA = torch.utils.data.DataLoader(all_datasetA, batch_size=batch_size, shuffle=False)
        all_loaderB = torch.utils.data.DataLoader(all_datasetB, batch_size=batch_size, shuffle=False)
    return train_loaderA, train_loaderB, all_loaderA, all_loaderB

def propose_sampling(z, mu, lam):
    D = z.shape[0]
    tmp_eta = np.zeros((K, D)); eta = np.zeros((D, K))
    for k in range(K): 
        tmp_eta[k] = np.diag(-0.5 * (z - mu[k]).dot(lam[k]).dot((z - mu[k]).T)).copy() 
        tmp_eta[k] += 0.5 * np.log(np.linalg.det(lam[k]) + 1e-7)
        eta[:, k] = np.exp(tmp_eta[k])
    eta /= np.sum(eta, axis=1, keepdims=True) 

    w = np.array([np.random.multinomial(1, eta[d]) for d in range(D)])
    w = np.argmax(w, axis=1)
    return w

def acceptance_rate(proposed_w, w, z, mu, lam):
    proposed_pdf = stats.multivariate_normal.pdf(z, mean=mu[proposed_w], cov=np.linalg.inv(lam[proposed_w]))
    current_pdf = stats.multivariate_normal.pdf(z, mean=mu[w], cov=np.linalg.inv(lam[w]))
    return np.min([1, proposed_pdf / current_pdf])

def sampling_mu_lambda(w, z):
    w = np.eye(K)[w]
    mu = np.zeros((K, latent_dim))
    lam = np.array([np.eye((latent_dim)) for _ in range(K)])
    for k in range(K):
        alpha_hat = np.sum(w[:, k]) + alpha
        m_hat = np.sum(w[:, k] * z.T, axis=1)
        m_hat += alpha * m
        m_hat /= alpha_hat
        tmp_w_dd_B = np.dot((w[:, k] * z.T), z)
        tmp_w_dd_B += alpha * np.dot(m.reshape(latent_dim, 1), m.reshape(1, latent_dim))
        tmp_w_dd_B -= alpha_hat * np.dot(m_hat.reshape(latent_dim, 1), m_hat.reshape(1, latent_dim))
        tmp_w_dd_B += np.linalg.inv(beta)
        w_hat = np.linalg.inv(tmp_w_dd_B)
        nu_hat = np.sum(w[:, k]) + nu
        
        # sampling \lambda^B and \mu^B
        lam[k] = wishart.rvs(size=1, df=nu_hat, scale=w_hat)
        mu[k] = np.random.multivariate_normal(mean=m_hat, cov=np.linalg.inv(alpha_hat * lam[k]), size=1).flatten()
    return mu, lam

def vae_train(dataset_name, agent, model_dir, iteration, train_loader, all_loader, prior_mu, prior_var):
    prior_mu = torch.from_numpy(prior_mu.astype(np.float32)).clone()
    prior_var = torch.from_numpy(prior_var.astype(np.float32)).clone()
    if dataset_name == "mnist":
        model = cnn_vae_module_mnist.VAE().to(device)
    elif dataset_name == "fruits360":
        model = cnn_vae_module_fruits.VAE().to(device)
        prior_mu = prior_mu.repeat_interleave(4, dim=0)
        prior_var = prior_var.repeat_interleave(4, dim=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = np.zeros((vae_iter))
    
    for i in range(vae_iter):
        model.train()
        train_loss = 0
        for batch_idx, (data, _, index) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, x_d = model(data)
            if iteration==0: 
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration)
            else:
                loss = model.loss_function(recon_batch, data, mu, logvar, prior_mu[index], prior_var[index], iteration=iteration)
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if i == 0 or (i+1) % 25 == 0 or i == (vae_iter-1):
            print('====> vae_iter: {} Average loss: {:.4f}'.format(i+1, train_loss / len(train_loader.dataset)))

        loss_list[i] = -(train_loss / len(train_loader.dataset))

    # plot
    plt.figure()
    plt.plot(range(0,vae_iter), loss_list, color="blue", label="ELBO")
    if iteration!=0: 
        loss_0 = np.load(model_dir+'/npy/loss'+agent+'_0.npy')
        plt.plot(range(0,vae_iter), loss_0, color="red", label="ELBO_I0")
    plt.xlabel('vae_iter'); plt.ylabel('ELBO'); plt.legend(loc='lower right')
    plt.savefig(model_dir+'/graph'+agent+'/vae_loss_'+str(iteration)+'.png')
    plt.close()

    np.save(model_dir+'/npy/loss'+agent+'_'+str(iteration)+'.npy', np.array(loss_list))
    torch.save(model.state_dict(), model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth")

    # send all latent
    z = []
    for batch_idx, (data, _, _) in enumerate(all_loader):
        data = data.to(device)
        _, _, _, z_d = model(data)
        z.append(z_d.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    return z
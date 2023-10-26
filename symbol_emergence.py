import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
from tool import calc_ari,cmx
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score as ari
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
from tool import visualize_gmm
from communication_field import CommunicationField
from agent import Agent
from utils import Dataset_mnist_index, create_dataloader


parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=50, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=50, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=10, metavar='K', help='number of category for GMM module')
parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--dataset-name', type=str, default="mnist", metavar='N', help='dataset name, mnist or fruits360')
parser.add_argument('--debug', type=bool, default=False, metavar='D', help='Debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print("CUDA",args.cuda)
if args.debug is True: args.vae_iter=2; args.mh_iter=2


############################## Making directory ##############################
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name
graphA_dir = "./model/"+file_name+"/graphA"; graphB_dir = "./model/"+file_name+"/graphB"
pth_dir = "./model/"+file_name+"/pth";npy_dir = "./model/"+file_name+"/npy"
reconA_dir = model_dir+"/"+file_name+"/reconA"; reconB_dir = model_dir+"/"+file_name+"/reconB"
log_dir = model_dir+"/"+file_name+"/log"; result_dir = model_dir+"/"+file_name+"/result"
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)
if not os.path.exists(pth_dir):    os.mkdir(pth_dir)
if not os.path.exists(graphA_dir):   os.mkdir(graphA_dir)
if not os.path.exists(graphB_dir):   os.mkdir(graphB_dir)
if not os.path.exists(npy_dir):    os.mkdir(npy_dir)
if not os.path.exists(reconA_dir):    os.mkdir(reconA_dir)
if not os.path.exists(reconB_dir):    os.mkdir(reconB_dir)
if not os.path.exists(log_dir):    os.mkdir(log_dir)
if not os.path.exists(result_dir):    os.mkdir(result_dir)

############################## Prepareing Dataset #############################
dataset_name = args.dataset_name
print("Dataset:",dataset_name)
if dataset_name == "mnist":
    pass
elif dataset_name == "fruits360":
    pass

train_loaderA, train_loaderB, all_loaderA, all_loaderB = create_dataloader(dataset_name, args.batch_size, args.category)

print(f"Total data:{len(all_loaderA.dataset)}, Category:{args.category}")
print(f"VAE_iter:{args.vae_iter}, Batch_size:{args.batch_size}")
print(f"MH_iter:{args.mh_iter}, MH_mode:{args.mode}(-1:Com 0:No-com 1:All accept)") 

############################## Define Agents ##############################
agentA = Agent(dataset_name, "A", dir_name, train_loaderA, all_loaderA)
agentB = Agent(dataset_name, "B", dir_name, train_loaderB, all_loaderB)

############################## Define Communication Field ##############################
communication_field = CommunicationField(agentA, agentB)
communication_field.symbol_emergence()
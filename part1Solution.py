import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.tensor(np.load('data/mixture_data.npz')['X'])
model_params = np.load('data/mixture_model.npz')
mu = torch.tensor(model_params['mu'])
Sigma = torch.tensor(model_params['Sigma'])
pi = torch.tensor(model_params['pi'])

def log_joint(x,mu,Sigma,pi):
    K,D = mu.shape
    x = x.reshape((1,1,D))
    SigmaInv = torch.inverse(Sigma)
    xmmu = x-mu[:,None,:]
    l = torch.log(pi).squeeze() -0.5*torch.logdet(2*np.pi*Sigma)
    -(0.5*xmmu@SigmaInv@torch.transpose(xmmu,1,2)).squeeze()
    return l
    
def posterior(x,mu,Sigma,pi):
    K = mu.shape[0]
    logpz = log_joint(x,mu,Sigma,pi)
    return torch.exp(logpz - torch.logsumexp(logpz,0))


def condition(x,mu,Sigma):
    K,D = mu.shape
    m = torch.nonzero(torch.isnan(x.flatten())).flatten()
    o = torch.nonzero(torch.logical_not(torch.isnan(x.flatten()))).flatten()
    x = x.reshape((1,1,D))
    SigmaooInv = torch.inverse(Sigma[:,o,:][:,:,o])
    xmmu = (x[:,:,o]-mu[:,None,o])
    xhats = mu[:,m] + (xmmu@SigmaooInv@Sigma[:,o,:][:,:,m]).squeeze()
    return(xhats)

def mixture_condition(x,mu,Sigma):
    K,D = mu.shape
    o = torch.nonzero(torch.logical_not(torch.isnan(x.flatten()))).flatten()
    pz = posterior(x[:,o],mu[:,o],Sigma[:,o,:][:,:,o])
    xhats = condition(x,mu,Sigma)
    xhat = pz.reshape((1,K))@xhats
    return(xhat)

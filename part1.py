import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.tensor(np.load('data/mixture_data.npz')['X'])
model_params = np.load('data/mixture_model.npz')
mu = torch.tensor(model_params['mu'])
Sigma = torch.tensor(model_params['Sigma'])
pi = torch.tensor(model_params['pi'])

def log_joint(X, mu, Sigma, pi, z):
    D = X.size(0)
    diff = X - mu[z, :].unsqueeze(0)
    inv_Sigma_z = torch.inverse(Sigma[z, :, :])
    temp = -0.5 * torch.sum(torch.matmul(diff, inv_Sigma_z) * diff, dim=1)
    
    return torch.log(pi[z]) - 0.5 * torch.logdet(Sigma[z, :, :]) - 0.5 * D * torch.log(torch.tensor(2 * torch.pi)) + temp

log_joints = [log_joint(data[0, :], mu, Sigma, pi, z) for z in range(10)]
for z, log_joint_val in enumerate(log_joints):
    print(f'log(P(X=x, Z={z})) = {torch.mean(log_joint_val):.4e}')

def log_sum_exp(x):
    max_val = torch.max(x, dim=1, keepdim=True)[0]
    return max_val + torch.log(torch.sum(torch.exp(x - max_val), dim=1, keepdim=True))

def conditional_probability(X, mu, Sigma, pi, z):
    term1 = log_joint(X, mu, Sigma, pi, z)
    term2 = log_sum_exp(torch.stack([log_joint(X, mu, Sigma, pi, z_prime) for z_prime in range(mu.shape[0])], dim=1))
    return torch.exp(term1 - term2)

conditional_probs = [conditional_probability(data[5, :], mu, Sigma, pi, z) for z in range(mu.shape[0])]
for z, prob in enumerate(conditional_probs):
    print(f'P(Z={z} | X=x) = {prob.item():.4e}')

def predict_right_half(X_l, mu, Sigma, pi):
    D_l = X_l.size(0)
    conditional_probs = [conditional_probability(X_l, mu[:, :D_l], Sigma[:, :D_l, :D_l], pi, z) for z in range(mu.shape[0])]
    predicted_right_halves = []
    for z in range(mu.shape[0]):
        term1 = mu[z, D_l:]
        term2 = torch.matmul(X_l - mu[z, :D_l], torch.matmul(torch.inverse(Sigma[z, :D_l, :D_l]),Sigma[z, :D_l, D_l:])) 
        predicted_right_half = term1 + term2
        predicted_right_halves.append(predicted_right_half)
    predicted_right_half = torch.zeros_like(predicted_right_halves[0])
    for z in range(mu.shape[0]):
        predicted_right_half += conditional_probs[z].squeeze(0) * predicted_right_halves[z]
    return predicted_right_half

plt.figure(figsize=(10, 4))
for i in range(10):
    D_l = int(data.shape[1] / 2)
    X_l = data[i, :D_l]
    X_r = predict_right_half(X_l, mu, Sigma, pi)
    X_lr = torch.cat((X_l, X_r))
    X_lr = X_lr.detach().numpy().reshape(28, 28)
    
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_lr, cmap="gray")
    plt.axis(False)

plt.savefig('outputImages.png')
plt.show()
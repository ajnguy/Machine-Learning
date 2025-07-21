import torch

def multi_class_loss(Y, G):
    N = Y.size(0)
    Y = Y.view(-1, 1)

    loss = -G.gather(1, Y) + torch.log(torch.sum(torch.exp(G), dim=1, keepdim=True))
    
    return loss.view(N, 1)




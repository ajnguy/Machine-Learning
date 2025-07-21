import torch
import torch.nn as nn
import torch.optim as optim

class linear(nn.Module):
    def __init__(self, K=100, optimizer=optim.Adam, num_epochs=1000, convergence_tol=0.0001, learning_rate=0.01):
        super(linear, self).__init__()
        self.encoder = nn.Linear(1024, K)
        self.decoder = nn.Linear(K, 2048)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.convergence_tol = convergence_tol
        self.criterion = nn.MSELoss()
        
        
    def f(self, x):
        return self.encoder(x)
    
    def g(self, x):
        return self.decoder(x)
    
    def loss(self, predicted, target):
        return self.criterion(predicted, target)

    def fit(self, L, AB):
        self.train()
        L = L.view(L.size(0), -1)
        prev_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            encoded = self.f(L)
            decoded = self.g(encoded)
            target_shape = decoded.size()[-1]
            target = AB.view(-1, target_shape)
            loss = self.loss(decoded, target)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}")
            
            if abs(loss - prev_loss) < self.convergence_tol:
                print("Convergence reached!")
                break

            prev_loss = loss
                
    def predict(self, L):
        self.eval()
        with torch.no_grad():
            L = L.view(L.size(0), -1)
            encoded = self.f(L)
            predicted_AB = self.g(encoded).view(L.size(0), 2048)
        return predicted_AB

    def save(self):
        torch.save({'model_state_dict': self.state_dict()}, "linear.pt")

    def load(self):
        checkpoint = torch.load("linear.pt")
        self.load_state_dict(checkpoint["model_state_dict"])

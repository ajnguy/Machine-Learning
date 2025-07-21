import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NeuralNetworkRegression:
    def __init__(self, input_dim, hidden_dim=64, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

    def f(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy()

    def risk(self, X, Y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
        Y_pred = self.model(X_tensor)
        return self.criterion(Y_pred, Y_tensor).item()

    def fit(self, X, Y, epochs=500):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            Y_pred = self.model(X_tensor)
            loss = self.criterion(Y_pred, Y_tensor)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        plt.plot(range(epochs), self.losses)
        plt.xlabel('Iteration Step')
        plt.ylabel('Risk Value')
        plt.title('Risk Value at each Iteration Step')
        plt.grid(True)
        plt.show()
        
        return self.model

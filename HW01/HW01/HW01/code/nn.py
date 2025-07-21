import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class CNNTimeSeriesRegressor:
    def __init__(self, window_size, lr=0.001):
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),  # compress time dimension
            nn.Flatten(),
            nn.Linear(64, 1)
        )

        self.model.apply(self.init_weights)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def f(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)  # (N, 1, window_size)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

    def risk(self, X_tensor, Y_tensor):
        Y_pred = self.model(X_tensor)
        return self.criterion(Y_pred, Y_tensor)

    def fit(self, X, Y, epochs):
        # Reshape input: (N, 1, window_size)
        X = X.reshape(-1, 1, X.shape[1])
        
        # Split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.4, random_state=42)

        # Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=128, shuffle=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        self.losses = []
        self.val_losses = []

        early_stop_patience = 500
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for X_batch, Y_batch in train_loader:
                self.optimizer.zero_grad()
                loss = self.risk(X_batch, Y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.model.eval()
            with torch.no_grad():
                val_loss = self.risk(X_val_tensor, Y_val_tensor)
            scheduler.step(val_loss.item())

            self.losses.append(avg_loss)
            self.val_losses.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Stopping early at epoch {epoch}: Best Validation Loss = {val_loss.item():.4f}")
                break

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss.item():.4f}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        plt.plot(self.losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_model_state

import torch
import torch.nn as NN
import torch.optim as optim
import matplotlib.pyplot as plt
import random

class nn(NN.Module):
    def __init__(self, K=100, optimizer=optim.Adam, num_epochs=1000, convergence_tol=0.00001, learning_rate=0.05):
        super(nn, self).__init__()
        self.encoder = NN.Sequential(
            NN.Conv2d(1, 64, kernel_size=3, padding=1),
            NN.ReLU(),
            NN.MaxPool2d(32, 32),
            NN.Flatten(start_dim=-1),
            
        )
        self.decoder = NN.Sequential(
            NN.Linear(1024, K),
            NN.ReLU(),
            NN.Linear(K, 2048),
            NN.Tanh()
        )
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.convergence_tol = convergence_tol
        self.criterion = NN.MSELoss()
    
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.f(x)
        #print(f"Encoder output shape: {x.shape}")
        x = torch.reshape(x, (1, 1024))
        x = self.g(x)
        #print(f"Decoder output shape: {x.shape}")
        return x
    
    def f(self, x):
        return self.encoder(x)
    
    def g(self, x):
        x = x.view(x.size(0), -1)
        return self.decoder(x)
    
    def loss(self, predicted, target):
        return self.criterion(predicted, target)
    
    def fit(self, L, AB, L_val, AB_val):
        prev_loss = float('inf')
        prev_val_loss = float('inf')
        learning_rate = self.optimizer.param_groups[0]['lr']
        learning_rate_patience = 2
        patience_counter = 0

        all_train_losses = []
        all_val_losses = []
        all_learning_rates = []
        for epoch in range(self.num_epochs):
            # Training
            self.train()
            total_loss = 0.0
            indices = list(range(len(L)))
            random.shuffle(indices)
            L_shuffled = L[indices]
            AB_shuffled = AB[indices]

            for start in range(0, len(L), 16):
                end = start + 16
                L_batch = L_shuffled[start:end]
                AB_batch = AB_shuffled[start:end]

                self.optimizer.zero_grad()

                AB_hat = self.forward(L_batch.unsqueeze(1))
                AB_hat = AB_hat.repeat(16, 1)
                #print("AB_hat: ", AB_hat.shape)
                #print("AB_batch: ", AB_batch.shape)
                
                loss = self.loss(AB_hat, AB_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / (len(L) // 16)

            all_train_losses.append(average_loss)
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(L_val), 16):
                    end = start + 16
                    L_val_batch = L_val[start:end]
                    AB_val_batch = AB_val[start:end]

                    AB_val_pred = self.forward(L_val_batch.unsqueeze(1))
                    AB_val_pred = AB_val_pred.repeat(16, 1)
                    
                    val_loss = self.loss(AB_val_pred, AB_val_batch)
                    val_losses.append(val_loss.item())

            average_val_loss = sum(val_losses) / len(val_losses)
            all_val_losses.append(average_val_loss)
            if average_val_loss < prev_val_loss:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= learning_rate_patience:
                learning_rate *= 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print(f"Reducing learning rate to {learning_rate}")
                patience_counter = 0
            all_learning_rates.append(self.optimizer.param_groups[0]['lr'])
            print(f"Epoch {epoch}/{self.num_epochs}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}")
            if abs(prev_loss - average_loss) < self.convergence_tol:
                print("Converged. Stopping training.")
                break
            prev_loss = average_loss
            prev_val_loss = average_val_loss
        self.plot_losses(all_train_losses, all_val_losses, all_learning_rates)

    def plot_losses(self, train_losses, val_losses, learning_rates):
        plt.plot(learning_rates,train_losses, label='Training Loss', color='blue')
        plt.plot(learning_rates, val_losses, label='Validation Loss', color='red')
        plt.title('Learning Rate vs Loss')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('learningRate_vs_loss.png')
        plt.show()
    
    def predict(self, L):
        self.eval()
        #print(L.shape)
        with torch.no_grad():
            additional_channels = torch.zeros(6, 32, 32)  
            augmented_L = torch.cat([L, additional_channels], dim=0)
            x = self.f(augmented_L.unsqueeze(1))
            x = torch.reshape(x, (1, 1024))
            predicted_AB = torch.reshape(self.g(x).repeat(10, 1), (L.size(0), 2048))
        return predicted_AB

    def save(self):
        torch.save({'model_state_dict': self.state_dict()}, "nn.pt")

    def load(self):
        checkpoint = torch.load("nn.pt")
        self.load_state_dict(checkpoint["model_state_dict"])
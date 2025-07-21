import numpy as np
import matplotlib.pyplot as plt
from regression import periodic_regression
from nn import CNNTimeSeriesRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset

#Load the data. Assumes the data directory is adjacent to the code directory.
# HW01
#  |-code
#  |-data
data = np.load("../data/data.npz")
Xtr = data["Xtr"]
Ytr = data["Ytr"]
Xte = data["Xte"]
Yte = data["Yte"]

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Ytr = scaler.fit_transform(Ytr)
Xte = scaler.fit_transform(Xte)
Yte = scaler.fit_transform(Yte)

#Define the component periods
K  = 9
rho=np.array([[12.42, 12.00, 12.66, 23.93, 25.82, 6.21, 4.14, 6.00, 6.10]]).T

# #Model instantiation example call
# model = periodic_regression()
# print(model.f)

# #Predict example call
# theta = np.zeros((2*K+1,1))
# Yhat  = model.f(theta,Xtr[:5,:],rho)
# print("Yhat =",Yhat.T)

# #Risk example call
# theta = np.zeros((2*K+1,1))
# risk  = model.risk(theta,Xtr,Ytr,rho)
# print("Risk =", risk)

# #Risk Gradient example call
# theta = np.zeros((2*K+1,1))
# riskGrad = model.riskGrad(theta,Xtr,Ytr,rho)
# print("Risk Gradient =",riskGrad.T)

# #Fit example call
# theta_opt = model.fit(Xtr,Ytr,rho)
# for i in range(theta_opt.shape[0]):
#     for j in range(theta_opt.shape[1]):
#         theta_opt[i, j] = "{:.4e}".format(theta_opt[i, j])
# print("Learned Parameters =",theta_opt.T)

# #Part f Average Squared Loss
# predicted_tide_heights = model.f(theta_opt, Xtr, rho).flatten()
# #print(predicted_tide_heights)
# train_loss = []
# for i in range(len(Ytr)):
#     train_loss.append((predicted_tide_heights[i] - Ytr[i])**2)
# train_avg_sqr_loss = np.mean(train_loss)

# test_loss = []
# for i in range(len(Yte)):
#     test_loss.append((predicted_tide_heights[i] - Yte[i])**2)
# test_avg_sqr_loss = np.mean(test_loss)

# print("Training set Average Squared loss: {:.4e}".format(train_avg_sqr_loss))
# print("Test set Average Squared loss: {:.4e}".format(test_avg_sqr_loss))

# #Part G First 48 Hours of Training 
# first_48_train = Ytr[:48].flatten()
# first_48_predicted = predicted_tide_heights[:48]
# first_48 = Xtr[:48].flatten()

# plt.scatter(first_48, first_48_train, color='black', label='Training Data')
# plt.plot(first_48, first_48_predicted)
# plt.xlabel('Hours')
# plt.ylabel('Tide Height')
# plt.title('Training Data vs Predicted Tide Heights (First 48 Hours)')
# plt.grid(True)
# plt.legend()
# plt.savefig('./Part_g_Training_vs_Prediction_first48.png')
# plt.show()

# #Part H Last 48 Hours of Test
# last_48_test = Yte[-48:].flatten()
# last_48_predicted = predicted_tide_heights[-48:]
# last_48 = Xte[-48:].flatten()

# plt.scatter(last_48, last_48_test, color='black', label='Test Data')
# plt.plot(last_48, last_48_predicted)
# plt.xlabel('Hours')
# plt.ylabel('Tide Height')
# plt.title('Test Data vs Predicted Tide Heights (Last 48 Hours)')
# plt.grid(True)
# plt.savefig('./Part_h_Test_vs_Prediction_last48.png')
# plt.show()

# Initialize CNN model
window_size = Xtr.shape[1]  
model = CNNTimeSeriesRegressor(window_size) 

# Predict example call
Yhat = model.f(Xtr[:5, :])
print("Yhat =", Yhat.T)

# Compute risk example call
risk = model.risk(torch.tensor(Xtr, dtype=torch.float32).reshape(-1, 1, window_size), 
                  torch.tensor(Ytr, dtype=torch.float32).reshape(-1, 1))
print("Risk =", risk.item())

# Train the model
model.fit(Xtr, Ytr, epochs=1000)

# Part f: Average Squared Loss
predicted_tide_heights = model.f(Xtr).flatten()  # f reshapes automatically
train_avg_sqr_loss = np.mean((predicted_tide_heights - Ytr.flatten())**2)

predicted_tide_heights_test = model.f(Xte).flatten()
test_avg_sqr_loss = np.mean((predicted_tide_heights_test - Yte.flatten())**2)

print("Training set Average Squared loss (CNN): {:.4e}".format(train_avg_sqr_loss))
print("Test set Average Squared loss (CNN): {:.4e}".format(test_avg_sqr_loss))

# Part g: First 48 Hours of Training
first_48_train = Ytr[:48].flatten()
first_48_predicted = predicted_tide_heights[:48]
first_48 = Xtr[:48].flatten()

plt.scatter(first_48, first_48_train, color='black', label='Training Data')
plt.plot(first_48, first_48_predicted)
plt.xlabel('Hours')
plt.ylabel('Tide Height')
plt.title('Training Data vs Predicted Tide Heights (First 48 Hours) (NN)')
plt.grid(True)
plt.legend()
plt.savefig('./Part_g_Training_vs_Prediction_first48_NN.png')
plt.show()

# Part h: Last 48 Hours of Test
last_48_test = Yte[-48:].flatten()
last_48_predicted = predicted_tide_heights_test[-48:]
last_48 = Xte[-48:].flatten()

plt.scatter(last_48, last_48_test, color='black', label='Test Data')
plt.plot(last_48, last_48_predicted)
plt.xlabel('Hours')
plt.ylabel('Tide Height')
plt.title('Test Data vs Predicted Tide Heights (Last 48 Hours) (NN)')
plt.grid(True)
plt.savefig('./Part_h_Test_vs_Prediction_last48_NN.png')
plt.show()


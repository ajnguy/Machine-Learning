import numpy as np
import matplotlib.pyplot as plt

#Load the data
data = np.load("../data/mixture_data.npz") #Load data
X = data["X"]

#Display the data
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i,:].reshape(28,28),cmap="gray")
    plt.axis(False)
plt.show()

#Load the model
model = np.load("../data/mixture_model.npz")
mu=model["mu"]
Sigma=model["Sigma"]
pi=model["pi"]

#Display the cluster means
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(mu[i,:].reshape(28,28),cmap="gray")
    plt.axis(False)
plt.show()

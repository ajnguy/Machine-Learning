import numpy as np
import torch 
import colorspaces
import matplotlib.pyplot as plt
from nn import nn
import torch.nn as NN
import torch.optim as optim
from sklearn.model_selection import train_test_split

data = np.load("../data/ae_data.npz") #Load data

#Extract train data
RGBtr = torch.tensor(data["RGBtr"])
Ltr   = torch.tensor(data["Ltr"])
ABtr  = torch.tensor(data["ABtr"])

#Extract test data
RGBte = torch.tensor(data["RGBte"])
Lte   = torch.tensor(data["Lte"])
ABte  = torch.tensor(data["ABte"])

# print("Ltr shape:", Ltr.shape)
# print("ABtr shape:", ABtr.shape)

#Validation Set
# Flatten Ltr
L_flat = Ltr.view(Ltr.size(0), -1)

# Split the data into training and validation sets without stratification
L_train_flat, L_val_flat, AB_train, AB_val = train_test_split(
    L_flat, ABtr.view(ABtr.size(0), -1), test_size=0.2, random_state=42
)

# Reshape the flattened tensors back to their original shapes
L_train = L_train_flat.view(L_train_flat.size(0), Ltr.size(1), Ltr.size(2))
L_val = L_val_flat.view(L_val_flat.size(0), Ltr.size(1), Ltr.size(2))
# print("L_train", L_train.shape)
# print("L_val", L_val.shape)
# print("AB_train", AB_train.shape)
# print("AB_val", AB_val.shape)

#Fit Model
nn = nn()
nn.fit(L_train, AB_train, L_val, AB_val)
nn.save()

#nn.load()

#Form random AB channel predictions
# pred_ABte = torch.rand(ABte.shape)-0.5
pred_ABte = nn.predict(Lte).view(-1,32,32,2)

#Combine with L channel
pred_LABte = torch.concat((Lte[:,:,:,None],pred_ABte),dim=3) 

#Plot true and predicted RGB images
plt.figure(figsize=(10,3))
for i in range(10):

    #Plot true image
    plt.subplot(3,10,i+1)
    plt.imshow(RGBte[i,:,:,:])
    plt.axis(False)

    #Plot L channel
    plt.subplot(3,10,i+11)
    plt.imshow(Lte[i,:,:],cmap="gray")
    plt.axis(False)

    #Convert LAB prediction to RGB and plot
    pred_RGBte = colorspaces.lab_to_rgb(pred_LABte[i,:,:,:])
    plt.subplot(3,10,i+21)
    plt.imshow(pred_RGBte[0,:,:,:])
    plt.axis(False)
plt.savefig("../nnOutput.png")
plt.show()
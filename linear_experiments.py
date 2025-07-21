import numpy as np
import torch 
import colorspaces
import matplotlib.pyplot as plt
from linear import linear
import torch.nn as nn
import torch.optim as optim

data = np.load("../data/ae_data.npz") #Load data

#Extract train data
RGBtr = torch.tensor(data["RGBtr"])
Ltr   = torch.tensor(data["Ltr"])
ABtr  = torch.tensor(data["ABtr"])

#Extract test data
RGBte = torch.tensor(data["RGBte"])
Lte   = torch.tensor(data["Lte"])
ABte  = torch.tensor(data["ABte"])

#Fit Model
linear = linear()

linear.fit(Ltr, ABtr)
linear.save()

#Form random AB channel predictions
# pred_ABte = torch.rand(ABte.shape)-0.5
pred_ABte = linear.predict(Lte).view(-1,32,32,2)

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
plt.savefig("../linearOutput.png")
plt.show()


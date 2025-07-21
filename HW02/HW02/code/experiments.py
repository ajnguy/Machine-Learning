import numpy as np
import matplotlib.pyplot as plt
from rlr import rlr

data = np.load("../data/data.npz")
Xtr = data["Xtr"] #Train inputs
Ytr = data["Ytr"] #Train labels
Ftr = data["Ftr"] #Train data file names

Xte = data["Xte"] #Test inputs
Yte = data["Yte"] #Test labels
Fte = data["Fte"] #Test data file names

#Set up example model and parameters
model = rlr()
N,D   = Xtr.shape
theta = np.zeros((D+1,1))
lam   = 1

#Example call to discriminant
out = model.discriminant(theta,Xtr[:2,:])
print("Discriminant:",out.T)

#Example call to predict
out = model.predict(theta,Xtr[:2,:])
print("Predict:",out.T)

#Example call to risk
out = model.risk(theta,Xtr[:2,:],Ytr[:2,:])
print("Risk:",out)

#Example call to regularizer
out = model.regularizer(theta)
print("Regularizer:",out)

#Example call to regularized risk
out = model.regularized_risk(theta,Xtr[:2,:],Ytr[:2,:],lam)
print("Regularized Risk:",out)

#Example call to risk_grad
out = model.risk_grad(theta,Xtr[:2,:],Ytr[:2,:])
print("Risk Gradient:",out.T)

#Example call to regularizer_grad
out = model.regularizer_grad(theta)
print("Regularizer Gradient:",out.T)

#Example call to regularized risk_grad
out = model.regularized_risk_grad(theta,Xtr[:2,:],Ytr[:2,:],lam)
print("Regularized Risk Gradient:",out.T)

#Example call to fit:
out=model.fit(Xtr[:2,:],Ytr[:2,:],lam)
print("Output of Fit:",out.T)

# Part i
lams = np.arange(0, 11)
training_errors = []
test_errors = []
for lam in lams:
    theta_opt = model.fit(Xtr, Ytr, lam)

    Ytr_pred = model.predict(theta_opt, Xtr)
    train_error = 1 - np.mean(Ytr_pred == Ytr)
    training_errors.append(train_error)

    Yte_pred = model.predict(theta_opt, Xte)
    test_error = 1- np.mean(Yte_pred == Yte)
    test_errors.append(test_error)

plt.plot(lams, training_errors, label='Training Error')
plt.plot(lams, test_errors, label='Test Error')
plt.xlabel('Regularization Parameter')
plt.ylabel('Error')
plt.legend()
plt.title('Training and Test Error vs. Regularization Parameter')
plt.savefig('parti-plot.png')
plt.show()

# Part j
lambdas = [0, 5, 10]
theta_0 = model.fit(Xtr, Ytr, 0)
theta_5 = model.fit(Xtr, Ytr, 5)
theta_10 = model.fit(Xtr, Ytr, 10)

features = [theta_0, theta_5, theta_10]

for i, lam in enumerate(lambdas):
    plt.stem(features[i])
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title(f'Model Parameters (lambda = {lam})')
    plt.savefig(f'partj-stemPlot-lam-{lam}.png')
    plt.show()

# Part k
lams = np.arange(0, 11)
def sparse_percentange(theta):
    num_sparse = np.sum(np.abs(theta) < .01)
    return (num_sparse / len(theta)) * 100
sparse_percentanges = []
for lam in lams:
    theta_opt = model.fit(Xtr, Ytr, lam)
    sparse_percentanges.append(sparse_percentange(theta_opt))

plt.plot(lams, sparse_percentanges, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Percentage of Sparse Weights')
plt.title('Sparsity of Weights vs Lambda')
plt.grid(True)
plt.savefig('partk-plot.png')
plt.show()

# Part l
lams = np.arange(0, 11)
training_errors = []
test_errors = []
for lam in lams:
    theta_opt = model.fit(Xtr, Ytr, lam)
    theta_opt = np.where(np.abs(theta_opt) < .01, 0, theta_opt)

    Ytr_pred = model.predict(theta_opt, Xtr)
    train_error = 1 - np.mean(Ytr_pred == Ytr)
    training_errors.append(train_error)

    Yte_pred = model.predict(theta_opt, Xte)
    test_error = 1 - np.mean(Yte_pred == Yte)
    test_errors.append(test_error)

plt.plot(lams, training_errors, label='Training Error')
plt.plot(lams, test_errors, label='Test Error')
plt.xlabel('Regularization Parameter')
plt.ylabel('Error')
plt.legend()
plt.title('Training and Test Error vs. Regularization Parameter (using sparsified model)')
plt.savefig('partl-plot.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

#Part a
x = np.arange(97)
K = 2
rho = [40, 20]
theta = [1.0, 0.5, 0.75, -0.25, 0.7]
b = theta[0]
w1 = theta[1]
w2 = theta[2]
phi1 = theta[3]
phi2 = theta[4]

f_theta = b + w1 * np.sin((2*np.pi/rho[0]) * x - phi1) + w2 * np.sin((2*np.pi/rho[1]) * x - phi2)

plt.plot(x, f_theta,)
plt.xlabel('x')
plt.ylabel('f_theta(x)')
plt.title("Prediction Function for f_theta(x)")
plt.grid(True)
plt.show()

#Part b
rho = [40, 20]
theta = [1.0, 0.5, 0.75, -0.25, 0.7]
b = theta[0]
w1 = theta[1]
w2 = theta[2]
phi1 = theta[3]
phi2 = theta[4]
D = [(20, -1), (50, 2.5)]
predicted_loss = []
for x, y in D:
    predicted_loss.append((b + w1 * np.sin((2*np.pi/rho[0]) * x - phi1) + w2 * np.sin((2*np.pi/rho[1]) * x - phi2) - y)**2)
empirical_risk = np.mean(predicted_loss)
print("Empirical Risk: ", empirical_risk)

#Part d
rho = [40, 20]
theta = [1.0, 0.5, 0.75, -0.25, 0.7]
b = theta[0]
w1 = theta[1]
w2 = theta[2]
phi1 = theta[3]
phi2 = theta[4]
D = [(20, -1), (50, 2.5)]
gradient_b = 0
gradient_w1 = 0
gradient_w2 = 0
gradient_phi1 = 0
gradient_phi2 = 0

for x, y in D:
    f_theta = b + w1 * np.sin((2*np.pi/rho[0]) * x - phi1) + w2 * np.sin((2*np.pi/rho[1]) * x - phi2)
    gradient_b += 2 * (f_theta - y) / 2
    gradient_w1 += 2 * (f_theta - y) * np.sin((2*np.pi/rho[0]) * x - phi1) / 2
    gradient_w2 += 2 * (f_theta - y) * np.sin((2*np.pi/rho[1]) * x - phi2) / 2
    gradient_phi1 += 2 * (f_theta - y) * -w1 * np.cos((2*np.pi/rho[0]) * x - phi1) / 2
    gradient_phi2 += 2 * (f_theta - y) * -w2 * np.cos((2*np.pi/rho[1]) * x - phi2) / 2

gradient = (gradient_b, [gradient_w1, gradient_w2], [gradient_phi1, gradient_phi2])
print("Gradient: ", gradient)

def riskGrad(theta,X,Y,rho): 
        #Complete this implementation
        K = 2
        # if len(theta.shape) == 1:
        #     theta = theta.reshape(-1, 1)
        thetaGrad = np.zeros((2*K + 1, 1))
        b = theta[0]
        w = theta[1:K+1]
        phi = theta[K+1:]
        gradient_b = 0
        gradient_w = np.zeros(K)
        gradient_phi = np.zeros(K)
        for x,y in zip(X, Y):
            temp = 0
            for i in range(K):
                temp += w[i] * np.sin((2*np.pi/rho[i]) * x - phi[i])
            f_theta = b + temp
                
            gradient_b += 2 * (f_theta - y) / len(X)
            for j in range(K):
                gradient_w[j] += 2 * (f_theta - y) * np.sin((2*np.pi/rho[j]) * x - phi[j]) / len(X)
                gradient_phi[j] += 2 * (f_theta - y) * -w[j] * np.cos((2*np.pi/rho[j]) * x - phi[j]) / len(X)
        thetaGrad[0, 0] = gradient_b
        for i in range(K):
            thetaGrad[i + 1, 0] = gradient_w[i]
        for j in range(K):
            thetaGrad[j + K + 1, 0] = gradient_phi[j]
        #print("riskGrad thetaGrad Shape: ", thetaGrad.shape)
        return(thetaGrad)

X = [item[0] for item in D]
Y = [item[1] for item in D]
print(riskGrad(theta, X, Y, rho))
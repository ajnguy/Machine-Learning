import numpy as np
from scipy.optimize import minimize, approx_fprime
import matplotlib.pyplot as plt

class periodic_regression:

    def __init__(self):
        pass

    def f(self,theta,X,rho):
        #Complete this implementation
        K = rho.shape[0]
        if len(theta.shape) == 1:
            theta = theta.reshape(2*K + 1, 1)
        #print("F Shape: ", theta.shape)
        theta = theta.flatten()
        Yhat = np.zeros(np.shape(X))
        b = theta[0]
        w = theta[1:K + 1]
        phi = theta[K + 1:]
        rho = rho.flatten()
        for i in range(X.shape[0]):
            temp = 0
            for j in range(len(w)):
                temp += w[j] * np.sin((2*np.pi/rho[j]) * X[i] - phi[j])
            Yhat[i, 0] = b + temp
        #print('F Yhat Shape: ', Yhat.shape)
        return(Yhat)

    def risk(self,theta,X,Y,rho):
        #Complete this implementation
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        #print("Risk Shape: ", theta.shape)
        risk = 0.0
        f_theta = self.f(theta, X, rho).flatten()
        predicted_loss = []
        for i in range(len(Y.flatten())):
            predicted_loss.append((f_theta[i] - Y[i])**2)
        risk = np.mean(predicted_loss)
        return(risk)

    def riskGrad(self,theta,X,Y,rho): 
        #Complete this implementation
        # def riskApprox(theta):
        #     return self.risk(theta, X, Y, rho)
        # print(approx_fprime(theta.flatten(), riskApprox, epsilon=1e-6))

        K = rho.shape[0]
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        thetaGrad = np.zeros(np.shape(theta))
        b = theta[0][0]
        w = theta[1:K+1].flatten()
        phi = theta[K+1:].flatten()
        rho = rho.flatten()
        X = X.flatten()
        Y = Y.flatten()
        gradient_b = 0
        gradient_w = np.zeros(K)
        gradient_phi = np.zeros(K)
        for x,y in zip(X, Y):
            temp = 0
            for i in range(K):
                temp += w[i] * np.sin((2*np.pi/rho[i]) * x - phi[i])
            f_theta = b + temp
                
            gradient_b += (f_theta - y) 
            for j in range(K):
                gradient_w[j] += (f_theta - y) * np.sin((2*np.pi/rho[j]) * x - phi[j]) 
                gradient_phi[j] += (f_theta - y) * -w[j] * np.cos((2*np.pi/rho[j]) * x - phi[j]) 
        gradient_b *= (2 / len(X))
        gradient_w *= (2 / len(X))
        gradient_phi *= (2/ len(X))
        thetaGrad[0, 0] = gradient_b    
        for i in range(K):
            thetaGrad[i + 1, 0] = gradient_w[i]
        for j in range(K):
            thetaGrad[j + K + 1, 0] = gradient_phi[j]
        #print("riskGrad thetaGrad Shape: ", thetaGrad.shape)
        return(thetaGrad)

    def fit(self,X,Y,rho):
        #Complete this implementation 
        K = rho.shape[0]
        theta = np.zeros((2*K+1,1))
        #print("Fit Theta: ", theta.shape)

        risks = []
        def callback(theta):
            theta = theta.reshape(-1, 1)
            risks.append(self.risk(theta, X, Y, rho))

        result = minimize(self.risk, theta, args=(X, Y, rho), method="L-BFGS-B", options={"disp":1}, tol=1e-6, 
                          callback=callback, jac=self.riskGrad)

        xvals = range(len(risks))
        plt.plot(xvals, risks)
        plt.xlabel('Iteration Step')
        plt.ylabel('Risk Value')
        plt.title('Risk Value at each Iteration Step')
        plt.grid(True)
        plt.savefig("./Part_d_riskValue_vs_iterationStep.png")
        plt.show()
        
        return(result.x.reshape((2*K+1,1)))
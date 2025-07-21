import numpy as np
import scipy.special 
from scipy.optimize import minimize, approx_fprime
import matplotlib.pyplot as plt

class rlr:

    def __init__(self):
        self.epsilon = .01
        pass

    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))      
       
    def discriminant(self,theta,X):
        #Complete this implementation
        N = X.shape[0]
        #print(X.shape)
        #print(theta.shape)
        g_theta = np.dot(X, theta[:-1]) + theta[-1]
        g_theta = g_theta.reshape(N, 1)
        return g_theta
    
    def predict(self,theta,X):
        #Complete this implementation
        N = X.shape[0]
        Y_hat = np.where(self.discriminant(theta, X) > 0, 1, -1) * 2 - 1
        #print(Y_hat)
        return Y_hat.astype(int)

    def risk(self,theta,X,Y):
        #Complete this implementation
        #print(Y)

        g_theta = self.discriminant(theta, X)
        # print(g_theta.shape, Y.shape)
        return np.sum(np.log(1 + np.exp(-Y * g_theta)))
        
    def regularizer(self,theta):
        #Complete this implementation
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        w = theta[:-1]
        return np.sum(np.sqrt(w**2 + self.epsilon))
        
    def regularized_risk(self,theta,X,Y,lam):
        #Complete this implementation
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        risk = self.risk(theta, X, Y)
        regularizer = self.regularizer(theta)
        return risk + lam*regularizer   
        
    def risk_grad(self,theta,X,Y): 
        #Complete this implementation
        # def Rapprox(theta):
        #     return self.risk(theta, X, Y)
        # approximate = approx_fprime(theta.flatten(), Rapprox, epsilon=1e-6)

        N,D = X.shape
        risk_gradient = np.dot(X.T, Y*(self.sigmoid(Y * self.discriminant(theta, X)) - 1)) / N
        bias_gradient = np.mean(Y*(self.sigmoid(Y * self.discriminant(theta, X)) - 1))
        risk_gradient = np.vstack((risk_gradient,bias_gradient))
        #print(risk_gradient.shape)

        # gradient_match = np.allclose(approximate, risk_gradient)
        # print(gradient_match)
        return risk_gradient
        
    def regularizer_grad(self,theta):
        #Complete this implementation
        # def Sapprox(theta):
        #     return self.regularizer(theta)
        # approximate = approx_fprime(theta.flatten(), Sapprox, epsilon=1e-6)
        
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        L,_ = theta.shape
        w = theta[:-1]
        regularizer_gradient = w / np.sqrt(w**2 + self.epsilon)
        regularizer_gradient = np.vstack((regularizer_gradient, 0.0))
        #print(regularizer_gradient.shape)

        # gradient_match = np.allclose(approximate, regularizer_gradient)
        # print(gradient_match)
        return regularizer_gradient

    def regularized_risk_grad(self,theta,X,Y,lam):
        #Complete this implementation
        # def RRapprox(theta):
        #     return self.regularized_risk(theta, X, Y, lam)
        # approximate = approx_fprime(theta.flatten(), RRapprox, epsilon=1e-6)
        N,D = X.shape
        regularized_risk_gradient = self.risk_grad(theta, X, Y) + lam * self.regularizer_grad(theta)
        #print(regularized_risk_gradient.shape)
        
        # gradient_match = np.allclose(approximate, regularized_risk_gradient)
        # print(gradient_match)
        return regularized_risk_gradient

    def fit(self,X,Y,lam):
        #Complete this implementation
        N,D = X.shape
        theta = np.zeros((D + 1, 1))

        result = minimize(self.regularized_risk, theta, args=(X, Y, lam), method="L-BFGS-B", options={"disp":1}, tol=1e-7, jac=self.regularized_risk_grad)
        result = result.x.reshape((D + 1, 1))
        #print(result.shape)

        return result
        

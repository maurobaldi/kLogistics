#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys

class Classifier:
    def __init__(self):
        self.name = "classifier"
        self.N = None
        self.M = None
        self.C = None
        self.w = None
        self.eta = None
        self.seed = None
        self.MAXITER = None
        self.c1 = None
        self.c2 = None
        
    def fit(self, X, t):
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.C = np.unique(t).size
        np.random.seed(self.seed)
        
    def predict(self, X):
        return None
    
    def exp(self, x):
        y = np.exp(x)
        return y
    
    def set_w(self, w):
        self.w = w
        
    def set_eta(self, eta):
        self.eta = eta
        
    def set_num_iterations(self, MAXITER):
        self.MAXITER = MAXITER
        
    def set_seed(self, seed):
        self.seed = seed

    def set_c1(self, c1):
        self.c1 = c1

    def set_c2(self, c2):
        self.c2 = c2
        
    def get_w(self):
        return self.w
    
    def get_Phi(self, X):
        Phi = np.insert(X, 0, 1.0, axis = 1)
        return Phi

    def get_eta(self, ii):
        c1 = self.c1
        c2 = self.c2
        if c1 is not None and c2 is not None:
            eta = c1/(ii + c2)
        else:
            eta = self.eta
        return eta


class Binary(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "binary classifier"
        
    def sigma(self, x):
        y = 1/(1 + self.exp(-x))
        return y
    
    def predict(self, X):
        Phi = self.get_Phi(X)
        Phi_w = Phi @ self.w
        y = self.sigma(Phi_w)
        t_hat = np.where(y >= .5, 1, 0)
        return t_hat
    
    def predictProb(self, X):
        Phi = self.get_Phi(X)
        Phi_w = Phi @ self.w
        y = self.sigma(Phi_w)
        return y

class Multi(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "multiclass classifier"
    
    def get_T(self, t):
        N = self.N
        C = self.C
        T = np.zeros([N, C])
        T[range(N), t] = 1
        return T
    
    def softmax(self, phi_n):
        v1 = self.w @ phi_n
        v2 = self.exp(v1)
        y = v2/v2.sum()
        return y
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Phi = self.get_Phi(X)
        N = X.shape[0]
        C = self.C
        predictions = np.zeros(N, dtype=int)
        for n in range(N):
            phi_n = Phi[n]
            y = self.softmax(phi_n)
            predictions[n] = np.argmax(y)
        return predictions
    
    def predictProb(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Phi = self.get_Phi(X)
        N = X.shape[0]
        C = self.C
        predictions = np.zeros(N, dtype=int)
        y_tot=[]
        for n in range(N):
            phi_n = Phi[n]
            y = self.softmax(phi_n)
            y_tot.append(y)
        return y_tot

class K:
    def __init__(self):
        self.k = None
        
    def set_k(self, k):
        self.k = k
        
    def get_k(self):
        return self.k
    
    def exp(self, x):
        k = self.k
        y = (np.sqrt(1 + k**2*x**2) + k*x)**(1/k)
        return y


class LogisticBinary(Binary):
    def __init__(self):
        super().__init__()
        self.name = "binary logistic classifier"
        
    def sigma(self, x):
        y = 1/(1 + self.exp(-x))
        return y
        
    def fit(self, X, t,verbose=0):
        super().fit(X, t)
        E_old = sys.float_info.max
        Phi = self.get_Phi(X)
        indexes = np.arange(self.N)
        
        for it in range(self.MAXITER):
            E = 0.0
            self.eta = self.get_eta(it)
            current_w=self.w
            np.random.shuffle(indexes)

            for n in indexes:
                phi_n = Phi[n, :]
                t_n = t[n]
                s = np.einsum("i,i->", self.w, phi_n)
                y_n = self.sigma(s)
                E_n = -t_n*np.log(y_n) - (1 - t_n)*np.log(1 - y_n)
                E+= E_n
                grad_w_E_n = (y_n - t_n)*phi_n
                self.w-= self.eta*grad_w_E_n
        
            if verbose:
                print("Iteration:", it, "E:", E)
                
            if E > E_old:
                self.w=current_w
                if verbose:
                    print("Uscita prematura!", it)
                break
                
            E_old = E
    

class KLogisticBinary(K, Binary):
    def __init__(self):
        super().__init__()
        self.name = "binary k-logistic classifier"
        
    def fit(self, X, t,verbose=0):
        super().fit(X, t)
        E_k_old = sys.float_info.max
        Phi = self.get_Phi(X)
        indexes = np.arange(self.N)   
        
        for it in range(self.MAXITER):   
            np.random.shuffle(indexes)      
            E_k = 0.0
            self.eta = self.get_eta(it)
            current_k=self.k
            current_w=self.w
            
            for n in indexes:
                phi_n = Phi[n, :]
                t_n = t[n]
                s = np.einsum("i,i->", self.w, phi_n)
                f = np.sqrt(1 + self.k**2*s**2)
                y_n = self.sigma(s)
                E_k_n = -t_n*np.log(y_n) - (1 - t_n)*np.log(1 - y_n)
                E_k+= E_k_n
                grad_w_E_n = (y_n - t_n)/f*phi_n
                grad_k_E_n = (y_n - t_n)/self.k*(s/f + np.log(self.exp(-s)))
                self.w-= self.eta*grad_w_E_n
                self.k-= self.eta*grad_k_E_n

            if verbose:
                print("Iteration:", it, "E_k:", E_k)
                
            if E_k > E_k_old:
                self.w=current_w
                self.k=current_k
                if verbose:
                    print("Uscita prematura!", it)
                break
            
            E_k_old = E_k


class KLogisticMulti(K, Multi):
    def __init__(self):
        super().__init__()
        self.name = "multi-class k-logistic classifier"
        
    def fit(self, X, t,verbose=0):
        super().fit(X, t)
        E_k_old = sys.float_info.max
        Phi = self.get_Phi(X)
        T = self.get_T(t)
        indexes = np.arange(self.N)
        
        for it in range(self.MAXITER):
            E_k = 0.0
            np.random.shuffle(indexes)
            self.eta = self.get_eta(it)
            current_k=self.k
            current_w=self.w            
            
            for n in indexes:
                phi_n = Phi[n]
                j = t[n]
                p = self.w@phi_n
                expkArray = self.exp(p)
                z = expkArray/expkArray.sum()
                E_k_n = -np.log(z[j])
                E_k+= E_k_n
                p1 = np.sqrt(1 + self.k**2*p**2)
                p2 = p/p1
                p3 = p2[j] - p2
                p4 = np.log(expkArray[j]/expkArray)
                p5 = p3 - p4
                p6 = expkArray*p5
                t_n = T[n]
                grad_coeff = (z - t_n)/p1
                grad_w_E_k_n = np.einsum("i,j", grad_coeff, phi_n)
                grad_k_E_k_n = -1/self.k*p6.sum()/expkArray.sum()
                self.w-= self.eta*grad_w_E_k_n
                self.k-= self.eta*grad_k_E_k_n
            
            if verbose: 
                print("Iteration:", it, "E_k:", E_k)
                
            if E_k > E_k_old:
                self.w=current_w
                self.k=current_k
                
                if verbose:
                    print("Uscita prematura!", it)
                break

            E_k_old = E_k   

class LogisticMulti(Multi):
    def __init__(self):
        super().__init__()
        self.name = "multi-class logistic classifier"
        
    def fit(self, X, t,verbose=0):
        super().fit(X, t)
        E_old = sys.float_info.max
        Phi = self.get_Phi(X)
        T = self.get_T(t)
        indexes = np.arange(self.N)
        
        for it in range(self.MAXITER):
            E = 0.0
            self.eta = self.get_eta(it)
            np.random.shuffle(indexes)
            current_w=self.w
            
            for n in indexes:
                phi_n = Phi[n]
                j = t[n]
                z = self.softmax(phi_n)
                E_n = -np.log(z[j])
                E += E_n
                t_n = T[n]
                grad_coeff = z - t_n
                grad_w_E_n = np.einsum("i,j", grad_coeff, phi_n)
                self.w-= self.eta*grad_w_E_n
                
            if verbose:
                print("Iteration:", it, "E:", E)
                
            if E > E_old:
                self.w=current_w
                if verbose:
                    print("Uscita prematura!", it)
                break

            E_old = E

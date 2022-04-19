#!/usr/bin/env python
# coding: utf-8
import numpy as np

def computePerformances(tTrue, tPred, classes):
    C = len(classes)
    TP = np.zeros(C)
    FP = np.zeros(C)
    FN = np.zeros(C)
    TN = np.zeros(C)
    PREC = np.zeros(C)
    REC = np.zeros(C)
    F1 = np.zeros(C)
    ACC = 0.0
    for i in range(C):
        c = classes[i]
        TP[i] = sum((tPred == c) & (tPred == tTrue))
        FP[i] = sum((tPred == c) & (tPred != tTrue))
        FN[i] = sum((tPred != c) & (tPred != tTrue))
        TN[i] = sum((tPred != c) & (tPred == tTrue))

        # Precision
        if TP[i] + FP[i] != 0:
            PREC[i] = TP[i]/(TP[i] + FP[i])
        else:
            PREC[i] = 0.0

        # Recall
        if TP[i] + FN[i] != 0:
            REC[i] = TP[i]/(TP[i] + FN[i])
        else:
            REC[i] = 0.0

        # F1
        if PREC[i] + REC[i] != 0:
            F1[i] = 2*PREC[i]*REC[i]/(PREC[i] + REC[i])
        else:
            F1[i] = 0.0

        # Accuracy
        ACC = 100*sum(tPred == tTrue)/len(tTrue)
    return ACC, PREC, REC, F1

# ## Superclass **Classifier**
class Classifier:
    def __init__(self):
        self.name = "Classifier"
        self.eta = None
        self.c1 = None
        self.c2 = None
        self.verbose = -1
        self.a = None
        self.b = None
        self.c = None
        self.d = None
            
    def set_c(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def set_W(self, W):
        self.W = W

    def get_W(self):
        W = self.W
        return W

    def set_eta(self, eta):
        self.eta = eta
        
    def get_eta(self, i):
        if self.eta is not None:
            eta = self.eta
        else:
            eta = self.c1/(i + self.c2)
        return eta
        
    def get_num_instances(self, X):
        N = X.shape[0]
        return N
        
    def get_num_features(self, X):
        D = X.shape[1]
        return D
    
    def get_num_classes(self, t):
        C = np.unique(t).size
        return C
    
    def get_T(self, t, N, C):
        T = np.zeros([N, C])
        T[range(N), t] = 1
        return T
        
    def get_Phi(self, X):
        Phi = np.insert(X, 0, 1, axis = 1)
        return Phi
    
    def get_array_phi(self, Phi):
        array_phi = np.einsum("in,nj->nij", Phi.T, Phi)
        return array_phi
    
    def get_cross_entropy(self, Y, T):
        E = -(T*np.log(Y)).sum()
        return E
    
    def exp(self, X):
        y = np.exp(X)
        return y

    def sigma(self, X):
        y = 1/(1 + self.exp(-X))
        return y
    
    def softmax_batch(self, Phi):
        N = Phi.shape[0]
        W = self.W
        P = Phi @ W.T        
        auxM = self.exp(P)
        auxSum = auxM.sum(axis = 1)
        Y = auxM/auxSum.reshape(N, 1)
        return Y, P

    def softmax_stochastic(self, phi_n):
        W = self.W
        v1 = W @ phi_n
        v2 = self.exp(v1)
        y = v2/sum(v2)
        return y
    
    def predict(self, X):
        Phi = self.get_Phi(X)
        auxTuple = self.softmax_batch(Phi)
        Y = auxTuple[0]
        t_hat = np.argmax(Y, axis = 1)
        return t_hat

    def predict_binary(self, X):
        Phi = self.get_Phi(X)
        Phi_w = Phi @ self.W       
        y = self.sigma(Phi_w)        
        t_hat = np.where(y >= .5, 1, 0)
        return t_hat
        
    def __str__(self):
        auxString = self.name + " " + "classifier" + "\n"
        if hasattr(self, "W") == True:
            auxString+= "W = " + np.array_str(self.W) + "\n"
        return auxString


# ## Child class **Logistic**
class Logistic(Classifier):    
    def __init__(self, n_iter = 40, eta = None, c1 = None, c2= None, tol = .01, l = 0, verbose = -1):
        super().__init__()
        self.name = "Logistic"
        self.n_iter = n_iter
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.l = l
        self.verbose = verbose
          
    def get_gradient(self, Phi, Y, T):
        DD = Y - T
        grad_partial_E = np.einsum("jn,nd->jd", DD.T, Phi)
        grad_E = grad_partial_E.reshape(grad_partial_E.size)
        return grad_E
                        
    def fit_gradient_batch(self, X, t, W = None):
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        C = self.get_num_classes(t)
        Phi = self.get_Phi(X)
        T = self.get_T(t, N, C)
        if W is None:
            w = np.zeros(C*(D + 1))
            self.W = w.reshape(C, -1)
        else:
            w = W.reshape(W.size)
            self.W = W
        w = w.astype(np.float64)
        E = np.Infinity
        i = 0
        STOP = False
        while STOP == False:            
            E_old = E
            Y, P = self.softmax_batch(Phi)            
            E = self.get_cross_entropy(Y, T)          
            eta = self.get_eta(i)
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E))
            grad_E = self.get_gradient(Phi, Y, T)
            grad_E/= np.linalg.norm(grad_E)
            grad_E+= self.l*w
            w-= eta*grad_E
            self.W = w.reshape(C, -1)
            i+= 1           
            #if np.abs(E - E_old) < self.tol or i == self.n_iter:
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E)
                STOP = True

    def fit_gradient_stochastic(self, X, t, W = None, seed = None):
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        C = self.get_num_classes(t)
        Phi = self.get_Phi(X)
        T = self.get_T(t, N, C)
        if W is None:
            self.W = np.zeros([C, D + 1])
        else:
            self.W = W
        self.W = self.W.astype(np.float64)
        rgen = np.random.RandomState(seed)
        E = np.Infinity
        i = 0
        STOP = False
        while STOP == False:
            E_old = E
            r = rgen.permutation(N)
            Phi_r = Phi[r]
            t_r = t[r]
            T_r = T[r]
            E = 0
            eta = self.get_eta(i)
            for phi_n, t_n, T_n in zip(Phi_r, t_r, T_r):
                y = self.softmax_stochastic(phi_n)
                d = y - T_n
                grad_partial = d.reshape(-1, 1)*phi_n.reshape(1, -1)
                self.W-= eta*grad_partial
                E_n = -np.log(y[t_n])
                E+= E_n
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E))
            i+= 1
            #if np.abs(E - E_old) < self.tol or i == self.n_iter:
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E)
                STOP = True

    def fit_gradient_stochastic_binary(self, X, t, W = None, seed = None):
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        Phi = self.get_Phi(X)
        if W is None:
            self.W = np.zeros(D + 1)
        else:
            self.W = W
        self.W = self.W.astype(np.float64)
        rgen = np.random.RandomState(seed)
        E = np.Infinity
        i = 0
        STOP = False
        while STOP == False:
            E_old = E
            r = rgen.permutation(N)
            Phi_r = Phi[r]
            t_r = t[r]
            E = 0
            eta = self.get_eta(i)
            for phi_n, t_n in zip(Phi_r, t_r):
                w_phi = np.dot(self.W, phi_n)
                y_n = self.sigma(w_phi)
                d = y_n - t_n
                grad_partial = d*phi_n
                self.W-= eta*grad_partial
                E_n = -(t_n*np.log(y_n) + (1 - t_n)*np.log(1 - y_n))
                E+= E_n
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E))
            i+= 1
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E)
                STOP = True

# ## Child class **KLogistic**
class KLogistic(Classifier):    
    def __init__(self, k = .5, n_iter = 40, eta = None, c1 = None, c2 = None,
                 tol = .01, l = 0, verbose = -1):
        super().__init__()
        self.name = "KLogistic"
        self.k = k
        self.n_iter = n_iter
        self.eta = eta
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.l = l
        self.verbose = verbose
        
    def exp(self, x):        
        k = self.k
        y = (np.sqrt(1 + k**2*x**2) + k*x)**(1/k)
        return y
            
    def get_gradient(self, Phi, Y, T, F, P):
        DD = Y - T
        G = F*DD
        grad_partial_E_k = np.einsum("jn,nd->jd", G.T, Phi)
        grad_E_k = grad_partial_E_k.reshape(grad_partial_E_k.size)
        return grad_E_k
            
    def fit_gradient_batch(self, X, t, W = None):
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        C = self.get_num_classes(t)
        Phi = self.get_Phi(X)
        T = self.get_T(t, N, C)
        if W is None:
            w = np.zeros(C*(D + 1))
            self.W = w.reshape(C, -1)
        else:
            w = W.reshape(W.size)
            self.W = W
        w = w.astype(np.float64)
        E_k = np.Infinity
        i = 0
        STOP = False
        while STOP == False:
            E_k_old = E_k
            Y, P = self.softmax_batch(Phi)
            E_k = self.get_cross_entropy(Y, T)
            eta = self.get_eta(i)
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E_k))
            F = 1/np.sqrt(1 + self.k**2*P*P)
            grad_E_k = self.get_gradient(Phi, Y, T, F, P)
            grad_E_k/= np.linalg.norm(grad_E_k)
            grad_E_k+= self.l*w
            w-= eta*grad_E_k
            self.W = w.reshape(C, -1)
            i+= 1
            #if np.abs(E_k - E_k_old) < self.tol or i == self.n_iter:
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E_k)
                STOP = True

    def softmax_stochastic(self, phi_n):
        W = self.W
        v1 = W @ phi_n
        v2 = self.exp(v1)
        y = v2/sum(v2)        
        return y, v1

    def fit_gradient_stochastic(self, X, t, W = None, seed = None):       
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        C = self.get_num_classes(t)
        Phi = self.get_Phi(X)
        T = self.get_T(t, N, C)
        if W is None:
            self.W = np.zeros([C, D + 1])
        else:
            self.W = W
        self.W = self.W.astype(np.float64)
        rgen = np.random.RandomState(seed)
        E = np.Infinity
        i = 0
        STOP = False
        while STOP == False:
            E_old = E
            r = rgen.permutation(N)
            Phi_r = Phi[r]            
            t_r = t[r]
            T_r = T[r]
            E = 0
            eta = self.get_eta(i)
            for phi_n, t_n, T_n in zip(Phi_r, t_r, T_r):
                y, auxArray = self.softmax_stochastic(phi_n)
                d = y - T_n
                grad_coeff = d/np.sqrt(1 + self.k**2*auxArray**2)
                grad_partial = grad_coeff.reshape(-1, 1)*phi_n.reshape(1, -1)
                self.W-= eta*grad_partial
                E_n = -np.log(y[t_n])                
                E+= E_n
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E))
            i+= 1
            #if np.abs(E - E_old) < self.tol or i == self.n_iter:
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E)
                STOP = True

    def fit_gradient_stochastic_binary(self, X, t, W = None, seed = None):
        N = self.get_num_instances(X)
        D = self.get_num_features(X)
        Phi = self.get_Phi(X)
        if W is None:
            self.W = np.zeros(D + 1)
        else:
            self.W = W
        self.W = self.W.astype(np.float64)
        rgen = np.random.RandomState(seed)
        E = np.Infinity
        i = 0
        STOP = False
        while STOP == False:
            E_old = E
            r = rgen.permutation(N)
            Phi_r = Phi[r]
            t_r = t[r]
            E = 0
            eta = self.get_eta(i)
            for phi_n, t_n in zip(Phi_r, t_r):
                w_phi = np.dot(self.W, phi_n)
                y_n = self.sigma(w_phi)
                d = y_n - t_n
                grad_coeff = d/np.sqrt(1 + self.k**2*w_phi**2)
                grad_partial = grad_coeff*phi_n
                self.W-= eta*grad_partial
                E_n = -(t_n*np.log(y_n) + (1 - t_n)*np.log(1 - y_n))
                E+= E_n
            if (self.verbose!= -1) and (i == 0 or (i + 1) % self.verbose == 0):
                print("Iteration %s, eta = %s, cross entropy = %s" % (i + 1, eta, E))
            i+= 1
            if i == self.n_iter:
                print("FINAL ITERATION, cross entropy = %s" % E)
                STOP = True

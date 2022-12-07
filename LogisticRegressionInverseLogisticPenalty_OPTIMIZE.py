import numpy as np
import pandas as pd 
import scipy.optimize as spo
from scipy.special import expit 
from scipy.sparse import diags

#Need to implement a "Scoring Method": using AUC or Precision on the full dataset. 
class LogisticRegressionInverseLogisticPenalty:
    """
    Class that penalizes the loss function using the inverse
    logistic function.  For minoirty class, use 1/logistic, 
    for majority class, use 1/(1-logistic).

    This version implements Scipy Unconstrained Optimization

    method: ['minority']

    Future release to incorporate other methods: 
    method: ['minority', 'majority', 'both'].  Default is 'minority'.  Originally developed
    to achieve better performance on imbalanced data. 

    The default solver is 'CG', but other standard solvers can be used.  See
    documentation for scipy.optimize.minimize at 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html 
    
    ###TO DO - 
    1) INCORPORATE THE ACCELERATION TERM, TAU
    2) CONSIDER CHECKING THE PENALTY ON THE MAJORITY CLASS - HARDER TO EXPLAIN, AND CHECK CONVEXITY
    3) 
    """
    def __init__(self, fit_intercept=True, method = 'minority', tau = 1, max_iter=1000, tol = 0.000001, solver = 'BFGS', random_beta_start = False, display = False):
        self.fit_intercept = fit_intercept
        self.method = method
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_beta_start = random_beta_start
        self.display = display
        self.bad_beta = []
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        #z = np.dot(X,beta)
        return expit(z)


    def __loss(self, beta, X, y): #Only penalize the Minority Class. 
        z = np.dot(X,beta)
        h = self.__sigmoid(z)

        if self.method == 'minority':
            return (-y*(self.tau/(h + 0.000001))*np.log(h+0.000001) - (1. - y)*np.log(1.-h+0.000001)).mean()
        elif self.method == 'majority':
            return (-y*np.log(h+0.000001) - 1./(1.-h)*(1. - y)*np.log(1.-h+0.000001)).mean()
        elif self.method == 'both':
            return (-y*(1./h)*np.log(h+0.000001) - 1./(1.-h)*(1. - y)*np.log(1.-h+0.000001)).mean()

    def __gradient(self, beta, X, y):
        #This error is happening quite a bit
        #Runtime Warning in FIT METHOD.  Check performance values here.
        # SPECIFIC MESSAGE: invalid value encountered in greater
        #All beta values are equal to nan ...
        #Query scipy.minimize why this might happen? Can we try different solvers? What about the tolerance? 
        z = np.dot(X,beta)
        if any(np.isnan(z)):
            print(f'\nCurrent Beta Values:\n {beta}\n')
            self.bad_beta = beta 

        h = self.__sigmoid(z)

        #Use the following code to deal with large values: 
        z[np.array([e > 700 if ~np.isnan(e) else False for e in z], dtype = bool)] = 700
        #out_vec[ np.array([e > 709 if ~np.isnan(e) else False for e in out_vec], dtype=bool) ] = 709
        #check for overflow: 
        if any(z > 700):
            z[np.array([e > 700 if (~np.isnan(e)| np.isinf(e)) else False for e in z], dtype = bool)] = 700
        if any(z < -700):
            z = np.where(z < -700, -100, z)

        if self.method == 'minority':
            return np.dot(X.T, (y*self.tau*np.exp(-z)*(np.log(h + 0.000001)-1.) + (1.-y)*h))/y.size
        elif self.method == 'majority':
            return np.dot(X.T, (-y*(1-h) + (1-y)*np.exp(z)*(1-np.log(1-h + 0.000001))))/y.size
        elif self.method == 'both':
            return np.dot(X.T, (y*np.exp(-z)*(np.log(h + 0.000001)-1)+ (1-y)*np.exp(z)*(1-np.log(1-h + 0.000001))))/y.size        

    def __hessian(self, beta, X, y): 
        z = np.dot(X,beta)
        if any(np.isnan(z)):
            print(f'\nCurrent Beta Values:\n {beta}\n')
            self.bad_beta = beta 

        #Issues with overflow in the np.exp function
        z[np.array([e > 700 if ~np.isnan(e) else False for e in z], dtype = bool)] = 700
        #out_vec[ np.array([e > 709 if ~np.isnan(e) else False for e in out_vec], dtype=bool) ] = 709
        #check for overflow: 
        if any(z > 700):
            z[np.array([e > 700 if (~np.isnan(e)| np.isinf(e)) else False for e in z], dtype = bool)] = 700
        if any(z < -700):
            z = np.where(z < -700, -100, z)

        h = self.__sigmoid(z)
        scale_left = self.tau*y*np.exp(-z)*(2-np.log(h+ 0.000001)+h)
        scale_right = (1-y)*h*(1-h)

        #Create diagonal matrix
        G = diags((scale_left + scale_right).flatten()/y.size)

        #Calculate hessian: 
        return X.T@G@X

    def get_params(self, deep=True):
        return {"fit_intercept": self.fit_intercept,   
                "method":self.method, "tau": self.tau,  
                "max_iter": self.max_iter,"tol": self.tol, 
                "solver":self.solver, "display":self.display}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            pred1 = self.__sigmoid(np.dot(X,self.beta))

        
        return np.stack((1-pred1,pred1),axis=1) 
        
    def predict(self, X, cut_off):
        return np.where(self.predict_proba(X) >= cut_off, 1, 0)
    
    #Need to add a score(X,y=None) method

    def fit(self, X, y):

        self.iterations = 1
        self.converged = False 
        self.hessian_inv = []
        self.msg = "Fit method called, but optimization not completed"

        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights/parameters initialization
        #beta = np.random.randint(-3,3, X.shape[1]).astype(float) #Think about randomizing this

        if self.random_beta_start: 
            self.beta = np.random.rand(X.shape[1])*2 -1
        else: 
            self.beta = np.zeros(X.shape[1])
        #z = np.dot(X, self.beta)
        #h = self.__sigmoid(z)


        ##Scipy Optimization - Implementation
        if self.solver in ['CG', 'BFGS', 'L-BFGS-B', 'SLSQP', 'TNC']:
            result = spo.minimize(fun = self.__loss, x0=self.beta, method = self.solver, jac = self.__gradient, args = (X, y.flatten()), tol = self.tol, options = {'maxiter':self.max_iter, 'disp':self.display})
        elif self.solver == 'Newton-CG':
            result = spo.minimize(fun = self.__loss, x0=self.beta, method = self.solver, jac = self.__gradient, hess= self.__hessian, args = (X, y.flatten()), tol = None, options = {'xtol': self.tol, 'maxiter':self.max_iter, 'disp':self.display})
        elif self.solver == 'trust-ncg':
            result = spo.minimize(fun = self.__loss, x0=self.beta, method = self.solver, jac = self.__gradient, hess= self.__hessian, args = (X, y.flatten()), tol = None, options = {'gtol':self.tol, 'maxiter':self.max_iter, 'disp':self.display})
        elif self.solver == 'Nelder-Mead': 
            result = spo.minimize(fun = self.__loss, x0=self.beta, method = self.solver, args = (X, y.flatten()), tol = self.tol, options = {'maxiter':self.max_iter, 'disp':self.display})
        else: 
            print("Please choose a method from the following:\nCG, BFGS, L-BFGS-B, Newton-CG, trust-ncg, SLSQP, Nelder-Mead, or TNC\nPlease ensure you use proper case.")

            return 

        if result.success:
            self.beta = result.x
            if self.solver == 'BFGS':
                self.hessian_inv = result.hess_inv 
        elif (result.success == False) and (result.message == 'Warning: Desired error not necessarily achieved due to precision loss.'):
            self.beta = result.x
            print("PRECISION LOSS - Desired error may not have been achieved.")
        else: 
            print("Optimization resulted in an error\n")
            self.beta = np.zeros(X.shape[1])
            print(result.message)

        self.iterations = result.nit    
        self.msg = result.message
        self.loss = result.fun
        self.converged = result.success
    


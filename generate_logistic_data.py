#From King/Zeng paper.  Monte Carlo experiments
import numpy as np


def simulate_logistic_data(size, beta0 = 1, beta1 = 1, mu_X = 0, sigma_X = 1, cov = 0): 
    """
    This function simulates data for a logistic regression 
    model with the specified values for the parameters 
    add number of predictors, then check array size of beta1
    """
    beta1 = np.array(beta1)
    beta0 = np.array(beta0)
    num_predictors = beta1.size

    #Create Covariance Matrix NOT CURRENTLY USED
    if len([sigma_X]) == 1:
        diag = np.repeat(sigma_X, num_predictors)
    else:
        diag = sigma_X
    cov = np.zeros((num_predictors, num_predictors))
    np.fill_diagonal(cov, diag)
    #Create X matrix
    X_int = np.ones(size).reshape(size,1)
    X_pred = mu_X + sigma_X*np.random.randn(num_predictors*size).reshape(size,num_predictors)
    X = np.concatenate((X_int, X_pred), axis = 1)
    
    #Create beta array
    Beta = np.hstack((beta0,beta1))
    XBeta = X.dot(Beta)

    #Create Probabilities
    probs = 1/(1+np.exp(-XBeta))
    Y = np.random.binomial(n=1, p=probs,size=size).reshape(size,)
    return X_pred,Y




"""
this script is the base script for redoing the Chapter 5 experiments to update the submission on the article submitted to the Journal of Classification.  Of note, we expand the cutoffs (0-->1 with step size 0.05) and also investigate the impact when N = 500_000 and 1_000_000.  

Each objective function (novel, regular, and balanced) is compared by averaging the performance metrics from 100 runs of 10-fold cross validation.

Create a couple lines of code to toggle between a local test and production on HPC
"""

import pandas as pd
import numpy as np
import time
import sys
from scipy.optimize.zeros import CONVERGED
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.exceptions import ConvergenceWarning
import warnings

#NOTE: Toggle for correct system paths
##LOCAL: 
# sys.path.append("C:\\Users\\trent\\OneDrive - Kennesaw State University\\KSU Courses\\Dissertation\\Dissertation\\python")
# from helper.model_utils_update import evaluate_performance_cutoff
# path_to_data = "Novel Algorithm\\Experiments CH5\\data2\\"
# path_to_results = "Novel Algorithm\\Experiments CH5\\results_JOC\\"

#HPC #####################
sys.path.append("/gpfs/SHPC_Data/home/tgeisler/ch5_sim_experiment/")
#os.chdir("/gpfs/SHPC_DATA/home/tgeisler/ch5_sim_experiment/")
path_to_data = "/gpfs/SHPC_Data/home/tgeisler/ch5_sim_experiment/data2/"
path_to_results = "/gpfs/SHPC_Data/home/tgeisler/ch5_sim_experiment/results_JOC/"
from model_utils_update import evaluate_performance_cutoff, remove_multicollinearity
#NOTE: End Note

import LogisticRegressionInverseLogisticPenalty_OPTIMIZE as myclf

#Change warnings to errors to catch in "try" block
warnings.filterwarnings('error')

###INITIALIZE EXPERIMENT PARAMETERS: 
row = 0
row_cutoff = 0
cutoff_list = np.arange(0,1,.05).round(2).tolist()
runs = 100
cv_folds = 10
newton_cg_tol = 0.0001 #Maybe try 0.0001
max_iter_start = 10000
data_suffix = "_JOC"
#Remove 'novel_ncg' due to performance issues with larger datasets
classifiers = ['novel_tnc', 'regular', 'balanced']

perf_cols = ['n', 'p', 'er', 'classifier', 'run', 'split', 'cutoff', 'fit_time', 'AUC', 'Avg_Prec', 'recall', 'precision', 'f1', 'fbeta_2', 'fbeta_point5', 'fit_error', 'converged','msg']
performance_df = pd.DataFrame(columns = perf_cols)
beta_coefficients = pd.DataFrame()

er_num = [0.0005, 0.001, 0.005, 0.01]
n = [50_000,100_000]
slopes = [8,16]
#Need for loop on datasets
#NOTE TESTING TESTING
# n =  [50_000]# 1_000_000
# nobs = 50_000
# slopes = [16, 32]
# slope = 16
# er_num = [0.0005, 0.001]
# er =  0.001
# runs = 1
# cv_folds = 2
#NOTE: END TESTING END TESTING

for nobs in n: 
    for slope in slopes:
        performance_df = pd.DataFrame(columns = perf_cols)
        beta_coefficients = pd.DataFrame() 
        
        for er in er_num:
            df = pd.read_csv(path_to_data + "ch5_data_n_" + str(nobs) + "slopes_"+str(slope)+"_event_rate_" + str(er) +".csv")

            X = np.array(df.filter(regex = 'Beta'))
            y = np.array(df['y'])
            mycols = ['Beta0'] + list(df.filter(regex='Beta').columns)
            ds = "n_" + str(nobs) + "_slopes_"+ str(slope)+"_er_" + str(er)

            for run in range(runs): #Implement the parallelized code here...
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=None)
                split = 0
                max_iter = max_iter_start

                for train, test in cv.split(X, y):
                    for classifier in classifiers:
                        #print(f'Run: {run}\nDataset with:\n{nobs} Observations\n{len(beta)} Slopes\n{er} Event Rate\n{classifier}')
                        clf_msg = 'None'
                        fit_error = 'None'
                        eval_error = 'None'
                        converged = True 

                        if classifier == 'novel_tnc': 
                            #try these methods "SLSQP", "L-BFGS-B", "Newton-CG", 'trust-ncg'
                            clf = myclf.LogisticRegressionInverseLogisticPenalty(tau = 1, solver = 'TNC', tol = newton_cg_tol, max_iter= max_iter_start, random_beta_start = False, method = 'minority')

                        elif classifier == 'novel_ncg':
                            clf = myclf.LogisticRegressionInverseLogisticPenalty(tau = 1, solver = 'Newton-CG', tol = newton_cg_tol, max_iter= max_iter_start, random_beta_start = False, method = 'minority')
                        elif classifier == 'regular': 
                            #solver options 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
                            clf = LogisticRegression(penalty = 'none', max_iter = max_iter_start)
                            #clf = LogisticRegression(penalty = 'none', max_iter = max_iter_start, solver = 'sag', n_jobs = -1)
                        elif classifier == 'balanced': 
                            clf = LogisticRegression(penalty = 'none', class_weight='balanced', max_iter = max_iter_start)
                            #clf = LogisticRegression(penalty = 'none', class_weight='balanced', max_iter = max_iter_start, solver = 'sag', n_jobs = -1)

                        try:
                            time_start = time.time() 
                            clf.fit(X[train], y[train])
                            time_stop = time.time()
                            if (classifier == 'novel_tnc') or (classifier == 'novel_ncg'):
                                converged = clf.converged
                                clf_msg = clf.msg
                                beta_coefficients.loc[row,'dataset'] = ds
                                beta_coefficients.loc[row,'er'] = er
                                beta_coefficients.loc[row,'classifier'] = classifier
                                beta_coefficients.loc[row,'run'] = run
                                beta_coefficients.loc[row,'split'] = split
                                for beta_num in range(len(mycols)):
                                    mybeta = mycols[beta_num]
                                    beta_coefficients.loc[row,mybeta] = clf.beta[beta_num]
                                beta_coefficients.loc[row,'converged'] = clf.converged
                            else: 
                                beta_coefficients.loc[row,'dataset'] = ds
                                beta_coefficients.loc[row,'er'] = er
                                beta_coefficients.loc[row,'classifier'] = classifier
                                beta_coefficients.loc[row,'run'] = run
                                beta_coefficients.loc[row,'split'] = split
                                for beta_num in range(len(mycols)):
                                    if beta_num == 0:
                                        beta_coefficients.loc[row,'Beta0'] = clf.intercept_[0]
                                    else: 
                                        mybeta = mycols[beta_num]
                                        beta_coefficients.loc[row, mybeta] = clf.coef_[0][beta_num-1]
                                beta_coefficients.loc[row,'converged'] = converged
                        except ConvergenceWarning as cw:
                            #What do we want to do here? Convergence warnings are not being thrown for balanced and regular methods.
                            fit_error = cw
                            print(f'CONVERGENCE WARNING FOR {classifier} Method.\nActual Warning: {cw}')
                            converged = False
                            
                            if max_iter >= 20000: 
                                max_iter = 20000
                            else: 
                                max_iter = 2*max_iter
                            clf.set_params(**{'max_iter':max_iter})
                            #beta_coefficients.loc[row,'converged'] = converged
                            print(f'Setting Max Iterations to: {max_iter}')

                            try: #Not sure what errors might be thrown here. 
                                for cutoff in cutoff_list:
                                    roc_auc, avg_prec, f1, fbeta, fbeta_point5, precision, recall, accuracy = evaluate_performance_cutoff(clf, X[test], y[test], cutoff = cutoff, beta = 2, verbose = False, dnn = False)                      
                                    fit_time = round(time_stop-time_start,3)
                                    cur_row = [nobs, slope, er, classifier, run, split, cutoff, fit_time, roc_auc,avg_prec,recall,precision,f1,fbeta,fbeta_point5,fit_error,converged, clf_msg]
                                    performance_df.loc[row_cutoff,:] = cur_row                       
                                    row_cutoff += 1 
                            except Exception as e2:
                                print(f'Exception Occured: {e2}')
                                continue

                        except Exception as e3:
                            fit_error = e3 #This currently isn't an issue when fitting our model. 
                            print(f"RANDOM EXCEPTION\nSPECIFIC MESSAGE: {e3}")
                            
                            if (classifier == 'novel_tnc') or (classifier == 'novel_ncg'):
                                print(f"\n Classifier Message: {clf.msg}")

                        else: #THIS IS THE SECTION OF CODE THAT GETS RUN when there is no exception caught in the try block.  Need a finally block to get code to execute on every iteration. 
                            for cutoff in cutoff_list:
                                roc_auc, avg_prec, f1, fbeta, fbeta_point5, precision, recall, accuracy = evaluate_performance_cutoff(clf, X[test], y[test], cutoff = cutoff, beta = 2, verbose = False, dnn = False)
                                
                                fit_time = round(time_stop-time_start,3)
                                cur_row = [nobs, slope, er, classifier, run, split, cutoff, fit_time, roc_auc,avg_prec,recall,precision,f1,fbeta,fbeta_point5,fit_error,converged, clf_msg]
                                performance_df.loc[row_cutoff,:] = cur_row                    
                                row_cutoff += 1
                        row += 1
                    split += 1
                if run % 10 == 0: 
                    performance_df.to_csv(path_to_results + "RESULTS_n_" + str(nobs) + "_p_" + str(slope) + data_suffix + ".csv", index = False)

            beta_coefficients.to_csv(path_to_results + "BETAS_n_" + str(nobs) + "_p_" + str(slope) + data_suffix + ".csv", index = False)
            performance_df.to_csv(path_to_results + "RESULTS_n_" + str(nobs) + "_p_" + str(slope) + data_suffix + ".csv", index = False)

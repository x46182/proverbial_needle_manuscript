#NOTE: DNN FUNCTIONS FOR NEURAL NETWORK EXPERIMENTS: 
import sys
import os
if 'tf' in sys.path[1]: 
    import tensorflow as tf
    from tensorflow import keras
    import keras.backend as kb
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, average_precision_score, roc_auc_score, recall_score, precision_score, f1_score, fbeta_score, balanced_accuracy_score
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
#from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import time
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
from itertools import product
from operator import itemgetter
import warnings

#py.sign_in('lzhang18', 'yk5o1duc8s')
#os.chdir("C:\\Users\\trent\\OneDrive - Kennesaw State University\\KSU Courses\\Dissertation\\Dissertation\\python")
#import LogisticRegressionWithLearnableLocalWeights as LR_local
#import LogisticRegressionWithLearnableLocalWeightsSWITCHED as switched
#import LogisticRegressionWithLearnableLocalWeights_TESTING as lrl

###FIX THIS IF USING WITH LOCAL LEARNABLE WEIGHTS...
def cv_roc(classifier, X, y, cv_splits=10, cv_seed=1):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    font = {"family": "Times New Roman", "size": 25}
    matplotlib.rc("font", **font)
    plt.figure(figsize=(10,10))

    i = 0

    for train, test in cv.split(X, y):
        if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
            classifier.fit(X[train], y[train], X[test], y[test])
            print("Fitting LR with Local Weights")
        else:
            print("Fitting LR without Local Weights")
            classifier.fit(X[train], y[train])

        probas_ = classifier.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#4D7EBF', label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='#282828', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + str(classifier.__class__.__name__) )
    plt.legend(loc="lower right", fontsize=12)
    plt.show()
    #plt.savefig(file_name, dpi=600)


def cv_auc(classifier, X, y, cv_splits=10, cv_seed=1):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    for train, test in cv.split(X, y):
        if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
            print("Fitting LR with Local Weights")
            classifier.fit(X[train], y[train], X[test], y[test])
            probas_ = classifier.predict_proba(X[test])
        else:
            print("Fitting LR without Local Weights")
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1
    #Why not take the mean of all the AUCs? 
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    return mean_auc

def cv_auc_runs_3splits(classifier, X, y, cv_splits=10, num_runs = 1, test_size = 0.2, cv_seed=None):
    """
    Use this function for the original LocalLearnable function from Lili    
    This function calculates the mean and standard
    deviation of the Area Under the ROC Curve on a test set after
    using k-fold cross-validation to train a model using a train/validation
    dataset.  The user defines the number of runs and the number of 
    cross-validation splits
    """
    aucs = []

    #Divide the data into train/validation and testing set.
    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                test_size = test_size, 
                                random_state = cv_seed,
                                stratify = y 
                                )

    for runs in range(0,num_runs):
        print("Working on Run Number: " + str(runs))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=None)
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        i = 0
        
        for train, test in cv.split(X_train, y_train):
            if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
                print("Fitting LR with Local Weights")
                #we use X_train[test] and y_train[test] as the validation set to help tune the
                #hyperparameters
                classifier.fit(X_train[train], y_train[train], X_train[test], y_train[test])
                #Use the actual testing set to calculate the performance
                probas_ = classifier.predict_proba(X_test)
            else:
                #This is not necessarily the "fair" way to do it.  A normal sklearn classifier
                #will never see the X_train[test] data.  All classifiers are getting evaluated to predict on the
                #same X_test data though.  Local learnable weights gets an opportunity to train with more data
                print("Fitting LR without Local Weights")
                probas_ = classifier.fit(X_train[train], y_train[train]).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print("The AUC for Fold " + str(i) + " and Run " + str(runs) + " is " + str(round(roc_auc,2)))
            print("\n The Current Mean AUC: \n" + str(np.mean(aucs)))
            print("\n The Current St Dev AUC: \n" + str(np.std(aucs)))
            i += 1
    #Look into the following commented code.  Seems like an odd way to get mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_odd = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
        
    return mean_auc, mean_auc_odd, std_auc, aucs


####This one is WRONG! However, this is the function I used to replicate Lili's results.
def cv_auc_runs_2splits(classifier, X, y, cv_splits=10, num_runs = 1, test_size = 0.2, cv_seed=None):
    """
    This function can be used on the UPDATED "LogisticRegressionWithLearnableLocalWeights"
    DO NOT USE ON ORIGINAL LEARNABLE LOGISTIC MODEL! (This function will Error)
    This function calculates the mean and standard
    deviation of the Area Under the ROC Curve on a test set after
    using k-fold cross-validation to train a model using a train/validation
    dataset.  The user defines the number of runs and the number of 
    cross-validation splits
    """
    aucs = []
    messages = []
    iterations = []

    for runs in range(0,num_runs):
        print("Working on Run Number: " + str(runs))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=None)
        #tprs = []
        #mean_fpr = np.linspace(0, 1, 100)
        i = 0
        
        for train, test in cv.split(X, y):
            if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
                print("Fitting ORIGINAL Learnable LR model - DEPRECATED")
                #we use X_train[test] and y_train[test] as the validation set to help tune the
                #hyperparameters
                clf = classifier
                clf.fit(X[train], y[train], X[test], y[test])
                #Use the actual testing set to calculate the performance
                probas_ = clf.predict_proba(X[test])
            else:
                print("Fitting Updated Learnable Model OR Standard Model in scikit learn")
                clf = classifier
                clf.fit(X[train], y[train])
                probas_ = clf.predict_proba(X[test])

                if clf.__class__.__name__=='LogisticRegressionWithLearnableLocalWeights2':
                    messages.append(clf.msg)
                    iterations.append(clf.iterations)

            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            #tprs.append(interp(mean_fpr, fpr, tpr))
            #tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print("The AUC for Fold " + str(i) + " and Run " + str(runs) + " is " + str(round(roc_auc,2)))
            print("\n The Current Mean AUC: \n" + str(np.mean(aucs)))
            print("\n The Current St Dev AUC: \n\n" + str(np.std(aucs)))
            i += 1
    
    #Look into the following commented code.  Seems like an odd way to get mean auc
    #mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    #mean_auc_odd = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
        
    return mean_auc, std_auc, aucs, messages, iterations

def search_for_optimal_weight_in_logistic_regression(ds_name, sample_size, event_rate, X, y, cv_splits=10, cv_seed=1):
    mean_auc_dict = {}
    std_auc_dict = {}

    for i in np.arange(0.0, 0.99, 0.01):
        class_0_weight = np.around(i, 2)
        class_1_weight = np.around(1-i,2)
        print( "computing the model with class 0 weight %f and class 1 weight %f" % (class_0_weight, class_1_weight))
        classifier = LogisticRegression(fit_intercept=True, solver="lbfgs", class_weight={0:class_0_weight, 1:class_1_weight})
        mean_auc, std_auc = cv_auc(classifier, X, y, cv_splits=cv_splits, cv_seed=cv_seed)
        mean_auc_dict[class_1_weight] = mean_auc
        std_auc_dict[class_1_weight] = std_auc
        maximum_key = max(mean_auc_dict, key=mean_auc_dict.get)
    mean_auc_pd = pd.DataFrame(mean_auc_dict.items(), columns=['Class 1 Weight', 'Mean AUC'])
    std_auc_pd = pd.DataFrame(std_auc_dict.items(), columns=['Class 1 Weight', 'Std AUC'])
    merged = mean_auc_pd.merge(std_auc_pd, on='Class 1 Weight')
    merged['Data Set'] = ds_name
    merged['Sample Size'] = sample_size
    merged['Event Rate'] = event_rate
    return merged


def experiment(ds_list):
    merged_list = []
    for ds_name, sample_size, event_rate, X, y in ds_list:
        start = time.time()
        print( "working on the data set %s with the sample size %i and event rate %f " % (ds_name, sample_size, event_rate))
        merged = search_for_optimal_weight_in_logistic_regression(ds_name, sample_size, event_rate, X, y, cv_splits=10)
        merged_list.append(merged)
        end = time.time()
        print( "running time %f" % (end - start))
    final_df = pd.concat(merged_list)
    return final_df

def Find_Optimal_Cutoff(target, predicted):
    """
    This function finds the optimal cutoff point for logistic regression. 
    The optimal cut off would occur where the tpr (true positive rate) is 
    high and the fpr (false positive rate) is low or near zero.  Naturally, 
    we can find the maximum of (tpr - (1-fpr)) at each of the threshold
    values.  This point corresponds to the point where the 
    true positive rate is approximately equal to (1-fpr).  Said another way, 
    this is where sensitivity is equal to specificity.

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0] 

# def evaluate(classifier, valid_x, valid_y): #be sure to input the "fitted" classifier
#     # generate the predicted probabilities
#     predictions_proba = classifier.predict_proba(valid_x)[:,1]

#     # find the optimal cutoff
#     cutoff = Find_Optimal_Cutoff(valid_y, predictions_proba)
#     print("Probability Cutoff: %.3f" % cutoff)
    
#     # make predictions based on the cutoff 
#     predictions = np.where(predictions_proba>=cutoff, 1, 0)
    
#     # Confusion Matrix
#     conf_matrix = confusion_matrix(valid_y, predictions)
#     plt.figure(figsize=(12.8,6))
#     sns.heatmap(conf_matrix, 
#             annot=True,
#             xticklabels=[0, 1], 
#             yticklabels=[0, 1],
#             cmap="Blues", fmt='g')
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.title('Confusion matrix')
#     plt.show()

#     # Accuracy
#     accuracy = accuracy_score(valid_y, predictions)
#     print("Accuracy: {:.1%}".format(accuracy))

#     # Type I Error and Type II Error
#     tn, fp, fn, tp = conf_matrix.ravel()
#     type_I_error = float(fp) / (fp + tn)
#     type_II_error = float(fn) / (fn + tp)
#     print("Type I Error: {:.1%}".format(type_I_error))
#     print("Type II Error: {:.1%}".format(type_II_error))


def evaluate_performance(classifier, valid_x, valid_y, verbose = False):
    # generate the predicted probabilities
    
    predictions_proba = classifier.predict_proba(valid_x)[:,1] 


    #generate area under the curve metric
    fpr, tpr, _ = roc_curve(valid_y, predictions_proba)
    roc_auc = auc(fpr, tpr) 
    
    # find the optimal cutoff
    cutoff = Find_Optimal_Cutoff(valid_y, predictions_proba)
    
    # make predictions based on the cutoff 
    predictions = np.where(predictions_proba>=cutoff, 1, 0)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(valid_y, predictions)

    # Accuracy
    accuracy = accuracy_score(valid_y, predictions)

    # Performance Metrics
    tn, fp, fn, tp = conf_matrix.ravel()

    if tp + fp != 0: 
        type_I_error = float(fp) / (fp + tn)
        type_II_error = float(fn) / (fn + tp)
        recall = float(tp) / (tp + fn) #also sensitivity
        specificity = float(tn) / (tn + fp)
        acc2 = float(tp + tn) / (tp + tn + fp + fn)
        precision = float(tp) / (tp + fp)
    else: 
        type_I_error = float(fp) / (fp + tn)
        type_II_error = float(fn) / (fn + tp)
        recall = float(tp) / (tp + fn) #also sensitivity
        specificity = float(tn) / (tn + fp)
        acc2 = float(tp + tn) / (tp + tn + fp + fn)
        precision = float('Nan')

    avg_prec = average_precision_score(valid_y, predictions_proba)

    try:
        f1 = 2*precision*recall/(precision + recall) 
    except RuntimeWarning:
        print("Runtime Warning Encountered in Evaluate Performance when Calculating F1.  Setting F1 to Zero")
        f1 = 0 
    except ZeroDivisionError:
        print("Division by Zero!  Setting F1 to Zero")
        f1 = 0

    if verbose is True: 
        print("Probability Cutoff: %.3f" % cutoff)
        print("Accuracy: {:.1%}".format(accuracy))
        print("{}: Type I Error: {:.1%}".format(classifier.__class__.__name__, type_I_error))
        print("{}: Type II Error: {:.1%}".format(classifier.__class__.__name__, type_II_error))
        print("{}: AUC: {:.1%}".format(classifier.__class__.__name__, roc_auc))
        print("{}: Precision: {:.1%}".format(classifier.__class__.__name__, precision))
        print("{}: Sensitivity: {:.1%}".format(classifier.__class__.__name__, recall))
        print("{}: f1: {:.1%}".format(classifier.__class__.__name__, f1))
        print("{}: Accuracy: {:.1%}".format(classifier.__class__.__name__, acc2))

    return roc_auc, recall, precision, f1, acc2, specificity, avg_prec

def evaluate_performance_cutoff(classifier, valid_x, valid_y, cutoff, beta=1, verbose = False, dnn = False):
    # generate the predicted probabilities
    #evaluate_performance_cutoff(model, X_test, y_test, cutoff = cutoff, beta = 2, verbose = False, dnn=True)
    if dnn == False: 
        predictions_proba = classifier.predict_proba(valid_x)[:,1] 
    else: 
        predictions_proba = classifier.predict(valid_x)


    #generate area under the curve metrics
    roc_auc = roc_auc_score(valid_y, predictions_proba) 
    avg_prec = average_precision_score(valid_y, predictions_proba)

    # find the optimal cutoff at cutoff=0
    if cutoff == 0: 
        cutoff = Find_Optimal_Cutoff(valid_y, predictions_proba)
    
    # make predictions based on the cutoff 
    predictions = np.where(predictions_proba>=cutoff, 1, 0).ravel()
    
    f1 = f1_score(valid_y, predictions)
    fbeta = fbeta_score(valid_y, predictions, beta = beta)
    fbeta_point5 = fbeta_score(valid_y, predictions, beta = 0.5)
    recall = recall_score(valid_y, predictions)
    precision = precision_score(valid_y, predictions, zero_division=0)
    accuracy = accuracy_score(valid_y, predictions)
    #gmean = geometric_mean_score(valid_y, predictions)


    #type_I_error = float(fp) / (fp + tn)
    #type_II_error = float(fn) / (fn + tp)


    return roc_auc, avg_prec, f1, fbeta, fbeta_point5, precision, recall, accuracy #, gmean

#Tuning the learning rate: 
# def tune_learning_rate(X, y, cv_splits=10, runs = 1, lr_grid = {'lr_theta':[.03,.01,.005], 'lr_lambda':[.01,.001,.0001]}, cv_seed = 1):
#     res_df = pd.DataFrame(columns = ["lr_theta", "lr_lambda", "Mean AUC"])

#     for lr_theta in lr_grid['lr_theta']:
#         for lr_lambda in lr_grid['lr_lambda']:
#             aucs = []

#             #For each cv_split fit the model and check the AUC.  Record the learning rate
#             #if the AUC is max
#             print("Working on Beta LR: {} and Lambda LR: {}".format(lr_theta,lr_lambda))
            
#             for run in range(runs): 
#                 print(f"Run: {run}")
#                 cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
#                 cv_seed = cv_seed + 1

#                 for train, test in cv.split(X, y):
#                     #print("Please be patient while we tune your weights :)")
#                     classifier = LR_local.LogisticRegressionWithLearnableLocalWeights(num_iter=100000,
#                                                                                         lr_theta=lr_theta, 
#                                                                                         lr_lambda=lr_lambda,
#                                                                                         fit_intercept=True, 
#                                                                                         verbose=False, 
#                                                                                         validate_w_auc = True, 
#                                                                                         test_auc_threshold = 1)
#                     classifier.fit(X[train],y[train],X[test],y[test])
#                     probas_ = classifier.predict_proba(X[test])
#                     # if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
#                     fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
#                     roc_auc = auc(fpr, tpr)
#                     aucs.append(roc_auc)
            
#             #Update Dataframe
#             res1 = pd.Series({"lr_theta":lr_theta,"lr_lambda":lr_lambda,"Mean AUC":np.mean(aucs)})
#             res_df = res_df.append(res1, ignore_index = True)


#     return res_df

#Tuning the learning rate: 
# def tune_learning_rate_test(X, y, cv_splits=2, lr_grid = {'lr_theta':[.03,.01,.005], 'lr_lambda':[.01,.001,.0001]}, 
#                             params = {'lr_theta': 0.01, 'lr_lambda': 0.01, 'num_iter': 100000, 'fit_intercept': True, 'verbose': False, 
#                             'perc_change_stop': 0.003, 'climbing_theta': False}, cv_seed = 42):
#     best_params = {}

#     for lr_theta in lr_grid['lr_theta']:

#         for lr_lambda in lr_grid['lr_lambda']:
#             aucs = []
#             #for each unique hyper-parameter combination, fit a new model and check AUC
#             cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)

#             #For each cv_split fit the model and check the AUC.  Record the learning rate
#             #if the AUC is max
#             for train, test in cv.split(X, y):
#                 print("Please be patient while we tune your weights :)")
#                 params['lr_theta'] = lr_theta
#                 params['lr_lambda'] = lr_lambda
#                 classifier = lrl.LogisticRegressionWithLearnableLocalWeights2()
#                 classifier.set_params(**params)
#                 classifier.fit(X[train],y[train])
#                 probas_ = classifier.predict_proba(X[test])
#                 # if classifier.__class__.__name__ == 'LogisticRegressionWithLearnableLocalWeights':
#                 fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#                 roc_auc = auc(fpr, tpr)
#                 print("AUC for current run:\n{}".format(round(roc_auc,2)))
#                 aucs.append(roc_auc)
            
#             #Update Dict
#             best_params[('lr_theta: '+ str(lr_theta), 'lr_lambda: ' + str(lr_lambda))] = np.mean(aucs)

#     return max(best_params, key = best_params.get), best_params

def lambda_stability(classifier, X, y, cv_splits=10, num_runs = 1, test_size = 0.2, cv_seed=None, method = "both"):
    """
        This function can be used on the UPDATED "LogisticRegressionWithLearnableLocalWeights"
        This function will record the lambda values after each call of the fit method for the 
        given classifier.  This will track the specific lambda value for each of the positive
        events.  

        return: dataframe where the index is the location of the positive events in the original
        dataset.  Each column will provide the lambda value for the given iteration.

        method = ["both", "minority", "majority"] 
    """
    aucs = []
    messages = []
    iterations = []

    #df to track lambda values
    if method == 'both':
        df = pd.DataFrame({"y": y})
        print("you assigned a df")
    elif method == 'minority': 
        df = pd.DataFrame(index = np.where(y > 0)[0])
    elif method == 'majority': 
        df = pd.DataFrame(index = np.where(y == 0)[0])
    else: 
        print("The method must be equal to both, minority, or both")
        return  

    for runs in range(0,num_runs):
        print("Working on Run Number: " + str(runs))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
        cv_seed += 1
        #track iterations within each run
        i = 0
        
        for train, test in cv.split(X, y):
            print("Fitting Updated Learnable Model OR Standard Model in scikit learn")
            clf = classifier
            clf.fit(X[train], y[train])
            probas_ = clf.predict_proba(X[test])

            #if clf.__class__.__name__=='LogisticRegressionWithLearnableLocalWeights2':
            messages.append(clf.msg)
            iterations.append(clf.iterations)
            
            if method == 'both': 
                temp = pd.Series(clf.final_lambda, index = train, name = "r" + str(runs) + "_i" + str(i))
            elif method == 'minority': 
                temp = pd.Series(clf.final_lambda, index = train[y[train] > 0], name = "r" + str(runs) + "_i" + str(i))
            elif method == 'majority': 
                temp = pd.Series(clf.final_lambda, index = train[y[train] == 0], name = "r" + str(runs) + "_i" + str(i))
            else: 
                print("Your method designation is wrong - use both, minority, majority")
                break 

            df = pd.concat([df,temp], axis = 1)
            print("You just added something to df...{}".format(df.head()))
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            #tprs.append(interp(mean_fpr, fpr, tpr))
            #tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print("The AUC for Fold " + str(i) + " and Run " + str(runs) + " is " + str(round(roc_auc,2)))
            print("\n The Current Mean AUC: \n" + str(np.mean(aucs)))
            print("\n The Current St Dev AUC: \n\n" + str(np.std(aucs)))
            i += 1
    
    #Look into the following commented code.  Seems like an odd way to get mean auc
    #mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    #mean_auc_odd = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
        
    return df, aucs, messages, iterations

def tune_classifier(classifier, X, y, grid, sort_by = 'auc', dict_return = "all", cv_splits=3, runs = 1, cv_seed = None, compare = False):
    """
    Tune a classifier given a dictionary of hyperparameters to iterate over
    Ensure the grid contains the appropriate parameters of the classifier
    The goal is to return the best parameters based on the desired performance metric

    sorting options include: "auc", "recall", "precision", "f1", "accuracy", "specificity", "average_precision"
    dict_return: "all", "top3", "top1"
    """
    #This is nice to be able to tune from a dictionary of unspecified items
    warnings.filterwarnings('error')
    keys = grid.keys()
    values = (grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    num_combinations = len(combinations)
    final_dict = combinations.copy()

    for combination_number in range(num_combinations):
        #initialize all the performance metric lists
        aucs = []
        recalls = []
        precisions = []
        f1s = []
        acc2s = []
        specificities = []
        avg_precs = []
        if compare is True: 
            reg_aucs = []
            reg_recalls = []
            reg_precisions = []
            reg_f1s = []
            reg_acc2s = []
            reg_specificities = []
            reg_avg_precs = []
        
        for run in range(runs):
            #for each unique hyper-parameter combination, fit a new model and check all performance
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
            cv_seed += 10
            clf = classifier

            if compare is True: 
                clf_reg = LogisticRegression(penalty = 'none', fit_intercept=True)

            clf.set_params(**combinations[combination_number])
            print("\nCurrently Working on: {}.  Run {}.\n".format(combinations[combination_number], run))
            #For each cv_split fit the model and check the AUC.  Record the learning rate
            #if the AUC is max
            for train, test in cv.split(X, y):
                try: 
                    clf.fit(X[train], y[train])
                    if clf.msg == "Max Iterations Reached":
                        print("Algorithm did not converge.  Max Iterations Reached.")
                except RuntimeWarning:
                    print("Runtime Warning Encountered.  Setting all performance values to 0")
                    roc_auc, recall, precision, f1, acc2, specificity, avg_prec = np.repeat(0,7)
                    next 
                else:
                    roc_auc, recall, precision, f1, acc2, specificity, avg_prec = evaluate_performance(clf, X[test], y[test], verbose = False)

                if compare is True:
                    #Consider checking errors here.
                    try: 
                        clf_reg.fit(X[train],y[train])
                    except Warning:
                        print('Warning on Regular Logistic Regression.  Most likely the result algorithm did not converge.  Setting performance values to zero!  Increasing maximum iterations by 50 and setting penalty = "L2" ')
                        max_iters = clf_reg.get_params()['max_iter']
                        clf_reg.set_params(**{'max_iter':max_iters+50, 'penalty':'l2'})
                        roc_auc2, recall2, precision2, f12, acc22, specificity2, avg_prec2 = np.repeat(0,7)
                    else: 
                        roc_auc2, recall2, precision2, f12, acc22, specificity2, avg_prec2 = evaluate_performance(clf_reg, X[test], y[test], verbose = False)
                    
                    
                    reg_aucs.append(roc_auc2)
                    reg_recalls.append(recall2)
                    reg_precisions.append(precision2)
                    reg_f1s.append(f12)
                    reg_acc2s.append(acc22)
                    reg_specificities.append(specificity2)
                    reg_avg_precs.append(avg_prec2)



                aucs.append(roc_auc)
                recalls.append(recall)
                precisions.append(precision)
                f1s.append(f1)
                acc2s.append(acc2)
                specificities.append(specificity)
                avg_precs.append(avg_prec)

        #Update Dict after cross validation for current parameter set
        #print the last set of Betas to get a sense of stability: 
        #print("\n Beta Coefficient on Last Run:\n {}".format(clf.beta))
        final_dict[combination_number]['n'] = len(aucs)
        final_dict[combination_number]["auc"] = np.mean(aucs)
        final_dict[combination_number]['Std AUC'] = np.std(aucs)
        final_dict[combination_number]["recall"] = np.mean(recalls)
        final_dict[combination_number]["precision"] = np.mean(precisions)
        final_dict[combination_number]["f1"] = np.mean(f1s)
        final_dict[combination_number]["accuracy"] = np.mean(acc2s)
        final_dict[combination_number]["specificity"] = np.mean(specificities)
        final_dict[combination_number]["average_precision"] = np.mean(avg_precs)
        #Add information about regular logistic regression
        if compare is True:
            final_dict[combination_number]["reg_auc"] = np.mean(reg_aucs)
            final_dict[combination_number]["reg_auc_std"] = np.std(reg_aucs)
            final_dict[combination_number]["reg_recall"] = np.mean(reg_recalls)
            final_dict[combination_number]["reg_precision"] = np.mean(reg_precisions)
            final_dict[combination_number]["reg_f1"] = np.mean(f1s)
            final_dict[combination_number]["reg_accuracy"] = np.mean(reg_acc2s)
            final_dict[combination_number]["reg_specificity"] = np.mean(reg_specificities)
            final_dict[combination_number]["reg_average_precision"] = np.mean(reg_avg_precs)

    #Return only top couple: 
    if dict_return == 'all':
        return sorted(final_dict, key = itemgetter(sort_by), reverse=True)
    elif dict_return == 'top3'and len(final_dict) >= 3:
        return sorted(final_dict, key = itemgetter(sort_by), reverse=True)[0:2]
    else:
        return sorted(final_dict, key = itemgetter(sort_by), reverse=True)[0]


def remove_multicollinearity(df, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = np.abs(df.corr())
    dataset = df.copy()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

def dataset_downsample(df, times = 0): #Use this in my Chapter 8 scripts for LR/DNN for real-world datasets
    majority = df[df.y == 0]
    minority = df[df.y ==1]
    for i in range(times): 
        minority = minority.sample(int(0.5*minority.shape[0]))
        
    return majority.append(minority, ignore_index = True)
############################################################################################################
############################################################################################################
###########################################################################################################

#Need to research how to create a custom loss function - this predicts the individual loss for each observation.  Should we do it by the entire batch? 

# import keras.backend as K

# def custom_mse(y_true, y_pred):
 
#     # calculating squared difference between target and predicted values 
#     loss = K.square(y_pred - y_true)  # (batch_size, 2)
    
#     # multiplying the values with weights along batch dimension
#     loss = loss * [0.3, 0.7]          # (batch_size, 2)
                
#     # summing both loss values along batch dimension 
#     loss = K.sum(loss, axis=1)        # (batch_size,)
    
#     return loss

#Creating custom metrics: 
# def mae(y_true, y_pred):
            
#     eval = K.abs(y_pred - y_true)
#     eval = K.mean(eval, axis=-1)
        
#     return eval
def novel_loss(y_true, y_pred):
    #possibly consider using this loss function with the added check now.  Stochastic gradient descent may work better. 
    epsilon = 0.00001
    #There is precedence for the epsilon - see Anand et al pg 962 which references Sussmann
    y_pred_new = tf.where(y_pred < 0.00001, 0.00001, y_pred)
    y_pred_new = tf.where(y_pred > .9999, .9999, y_pred) 
    minority_obs = y_true > 0
    minority_loss = -tf.math.log(y_pred_new + epsilon)/(y_pred_new + epsilon)
    majority_loss = -tf.math.log(1-y_pred_new + epsilon)
    return tf.where(minority_obs, minority_loss, majority_loss)

def novel_loss2(y_true, y_pred):
    epsilon = 0.00001
    y_pred_new = tf.where(y_pred < 0.000001, 0.000001, y_pred)
    y_pred_new = tf.where(y_pred > .9999, .9999, y_pred) 
    minority_obs = y_true > 0
    minority_loss = -tf.math.log(y_pred_new + epsilon) \
                    /(y_pred_new + epsilon)
    majority_loss = -tf.math.log(1-y_pred_new + epsilon)
    loss_obs = tf.where(minority_obs, minority_loss, majority_loss)
    return kb.mean(loss_obs, axis=-1)

def focal_loss(y_true, y_pred):
    #Also consider adding an \epsilon in tf.math.log function.  We were experiencing overflow issues without the tf.where check.
    epsilon = 0.00001
    gamma = 2
    minority_obs = y_true > 0
    minority_loss = -((1-y_pred)**gamma) * tf.math.log(y_pred + epsilon)
    majority_loss = -(y_pred**gamma) * tf.math.log(1-y_pred + epsilon)
    loss_obs = tf.where(minority_obs, minority_loss, majority_loss)
    return kb.mean(loss_obs, axis=-1)

def build_model(input_dim, layers = 2, nodes_per_layer = 16, batch_normal = False, activation = 'elu', kernel_initializer='lecun_normal', seed = 42): 
    model = None 
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_dim))

    if kernel_initializer=='lecun_normal': 
        initializer = tf.keras.initializers.LecunNormal(seed = seed)
    if kernel_initializer == 'he_normal': 
        initializer = tf.keras.initializers.HeNormal(seed = seed) #add additional intializers as necessary. 
    for _ in range(layers):     
        model.add(keras.layers.Dense(nodes_per_layer,
                                     activation = activation, 
                                     kernel_initializer= initializer))
        if batch_normal is True: 
            model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(1, activation = "sigmoid"))

    return model

def get_run_logdir(root_logdir, model, beta, nobs, er, lr): 
    import time
    run_id = time.strftime(model + "_" + str(beta) + "_" + str(er)[2:] + "_" + str(nobs) + "_LR_" + str(lr)[2:] +  "_run_%H%M%S")
    return os.path.join(root_logdir, run_id)

def get_run_logdir2(root_logdir, model, lr): 
    import time
    run_id = time.strftime(model + "_" + str(lr) + "_run_%H%M%S")
    return os.path.join(root_logdir, run_id)

def get_run_logdir_ch8(root_logdir, model, dataset, downsample, batch_size, lr, run): 
    import time
    run_id = time.strftime(model + "_" + dataset + "_d" + str(downsample) + "_bs" + str(batch_size) + "_" + str(lr)  + "_R" + str(run))
    return os.path.join(root_logdir, run_id)

def compile_fit_model(X_train, y_train, X_val, y_val, model, model_type, epochs = 400, lr = 0.0005, patience = 10, monitor = 'val_loss', optim = 'RMSprop', batch_size = 32, novel_cbw = True, parallel = False, tensorboard = False, logdir = ".\\Novel Algorithm\\Experiments CH6\\tf_logs\\test"): 
    class_wt = None
    loss = 'binary_crossentropy'
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor = monitor, patience = patience, min_delta=0.0, restore_best_weights=True)
    cbs = [early_stopping_cb]
    if tensorboard is True: 
        tb_cb = keras.callbacks.TensorBoard(logdir)
        cbs = [early_stopping_cb, tb_cb]
        
    if model_type == 'reg': 
        print(f'Adjusting for {model_type}')

    if model_type == 'novel': 
        print(f'Adjusting for {model_type}')
        loss = novel_loss2

        if novel_cbw is True: 
            weights = y_train.size/(2*np.bincount(y_train.astype(int)))
            class_wt = {0:weights[0], 1:weights[1]}

    if model_type == 'focal': 
        print(f'Adjusting for {model_type}')
        loss = focal_loss

        if novel_cbw is True: 
            weights = y_train.size/(2*np.bincount(y_train.astype(int)))
            class_wt = {0:weights[0], 1:weights[1]}

    if model_type == 'balanced':
        weights = y_train.size/(2*np.bincount(y_train.astype(int)))
        print(f'Adjusting for {model_type}\nW0: {weights[0]}\nW1: {weights[1]}') 
        class_wt = {0:weights[0], 1:weights[1]}
    #Set optimizer: See page 359 in Geron Hands2 for convergence speed and quality.
    if optim == 'adam':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999) #default lr= 0.01
    elif optim == 'SGD':
        optimizer = keras.optimizers.SGD(lr = lr, momentum = 0.9, nesterov=True) #default lr = 0.01
    else: 
        optimizer =  keras.optimizers.RMSprop(lr = lr, rho = 0.9) #default lr= 0.001

    model.compile(loss = loss,
                optimizer = optimizer,
                metrics = ['AUC', 'Precision', 'TruePositives']) 
                # metrics = [tf.metrics.AUC(curve = 'PR'), 
                #            tf.metrics.Precision(), 
                #            tf.metrics.TruePositives()])
    
    model_hist = model.fit(X_train, 
                        y_train, 
                        epochs = epochs, 
                        #validation_split = val_split,
                        validation_data = (X_val, y_val),
                        callbacks = cbs, 
                        class_weight = class_wt,
                        batch_size = batch_size, 
                        verbose = 0,
                        use_multiprocessing = parallel)
    
    model_hist = pd.DataFrame(model_hist.history)

    return model, model_hist
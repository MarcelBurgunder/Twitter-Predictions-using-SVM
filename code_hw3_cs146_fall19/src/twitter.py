"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list

        string = fid.read()
        listt = (extract_words(string))

        unique_listt = []
        for i in listt: 
            if i not in unique_listt: 
                unique_listt.append(i)

        for j in range(len(unique_listt)):
            word_list.update({j : unique_listt[j]})

        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        tweets = fid.read()
        for i in range(num_words):
            j = 0   # line counter
            for line in tweets.split("\n"):
                target = word_list[i]
                search = extract_words(line)
                if (target in search):
                    feature_matrix[j][i] = 1
                j+=1


                # print(word_list.get(i))
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    if metric == "f1-score" or metric == "F1-Score":
        return metrics.f1_score(y_true, y_pred)
    if metric == "auroc" or metric == "AUROC":
        return metrics.roc_auc_score(y_true, y_pred)
    else:
        print("ERROR - metric undefined")
        return 0
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance

    sum_of_performances = 0
    k = kf.get_n_splits(X, y)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index] # generate train and test data
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_true = y_test
        y_pred = clf.predict(X_test)
        sum_of_performances += performance(y_true, y_pred, metric)

    average_performance = sum_of_performances/k
    return average_performance

    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    C_array = [0.001, 0.01, 0.1, 1, 10, 100]
    min_performance = 0
    C = 0
    for c_potential in C_array:
        clf = SVC(kernel='linear', C=c_potential)
        potential_performance = cv_performance(clf, X, y, kf, metric=metric)
        print("Performance based on " + str(metric) + " for C = " + str(c_potential) + " is " + str(potential_performance))
        if min_performance < potential_performance:
            C = c_potential

    return C
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.predict(X)
    y_true = y        
    return performance(y_true, y_pred, metric=metric)
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc"]
    
    ### ========== TODO : START ========== ###
    # part 1: split data into training (training + cross-validation) and testing set

    X_train = X[0:560, :]
    X_test = X[560:630, :]
    y_train = y[0:560]
    y_test = y[560:630]

    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
    
    # part 2: create stratified folds (5-fold CV)

    kf = StratifiedKFold(n_splits = 5)
    kf.get_n_splits(X_train, y_train)
    
    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV

    C_optimal_accuracy = select_param_linear(X_train, y_train, kf, metric="accuracy")
    C_optimal_f1_score = select_param_linear(X_train, y_train, kf, metric="F1-Score")
    C_optimal_auroc = select_param_linear(X_train, y_train, kf, metric="AUROC")
        
    # part 3: train linear-kernel SVMs with selected hyperparameters

    clf_max_accuracy = SVC(kernel='linear', C=C_optimal_accuracy)
    clf_max_f1_score = SVC(kernel='linear', C=C_optimal_f1_score)
    clf_max_auroc = SVC(kernel='linear', C=C_optimal_auroc)

    clf_max_accuracy.fit(X_train, y_train)
    clf_max_f1_score.fit(X_train, y_train)
    clf_max_auroc.fit(X_train, y_train)
    
    # part 3: report performance on test data

    perf_test_accuracy = performance_test(clf_max_accuracy, X_test, y_test, metric="accuracy")
    perf_test_f1_score = performance_test(clf_max_f1_score, X_test, y_test, metric="f1-score")
    perf_test_auroc = performance_test(clf_max_auroc, X_test, y_test, metric="auroc")

    print("For the test data:")

    print("Trained classifier with optimal C (" + str(C_optimal_accuracy) + ") for accuracy achieved a performance of: " + str(perf_test_accuracy))
    print("Trained classifier with optimal C (" + str(C_optimal_f1_score) + ") for F1-Score achieved a performance of: " + str(perf_test_f1_score))
    print("Trained classifier with optimal C (" + str(C_optimal_auroc) + ") for AUROC achieved a performance of: " + str(perf_test_auroc))

    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()

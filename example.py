from preprocessing import TextPreparation
from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod
from methods.naive_bayes import NaiveBayesMethod
from methods.reccurent_neural import ReccurentNeuralMethod
from methods.sklearn_neural import SKlearnNeural
from methods.voting_classifier import VoitingClassifierMethod
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from sklearn.preprocessing import scale


def divide_dataset(test_size):
    x, y = TextPreparation.get_dataset_new()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    print(len(X_train[0]), len(X_train[1]))
    return X_train, X_test, y_train, y_test

def prepare_text(blocks_size=100, max_features=500):
    ex = TextPreparation(max_features=max_features)
    # ex.prepare_all_files()
    # ex.create_train_keras()
    # ex.get_dataset_keras()
    ex.create_train()
    # ex.create_all_dicts(blocks_size=blocks_size)

def prepare_text2(max_features=500):
    ex = TextPreparation(max_features=max_features)
    ex.prepare_all_files(authors_number=3, files_number=4, file_length=20000)
    # ex.create_train_keras()
    # ex.get_dataset_keras()
    ex.create_train(authors_number=3, files_number=4)
    # ex.create_all_dicts(blocks_size=blocks_size)

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_solvers = ["newton-cg", "sag", "saga"]
    # solver = "newton-cg"
    for solver in lr_solvers:
        print("-----------------LOGISTIC REGRESSION: {}------------------------".format(solver))
        # lr = LogisticRegressionMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
        #                               np.array(y_train).astype("float32"), np.array(y_test).astype("float32"),
        #                               solver=solver)
        lr = LogisticRegressionMethod(X_train, X_test, y_train, y_test)
        lr.do_LR()

def random_forests(X_train, X_test, y_train, y_test):
    print("-----------------RANDOM FORESTS------------------------")
    rf = RandomForestMethod(X_train, X_test, y_train, y_test)
    rf.do_RF()

def svm(X_train, X_test, y_train, y_test):
    print("-----------------SVM METHOD------------------------")
    svm = SVMMethod(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
    svm.do_SVM()

def knn(X_train, X_test, y_train, y_test):
    print("-----------------KNN------------------------")
    knn = KNNMethod(X_train, X_test, y_train, y_test)
    knn.do_knn()

def naive_bayes(X_train, X_test, y_train, y_test):
    print("-----------------NAIVE BAYES------------------------")
    bayes = NaiveBayesMethod(X_train, X_test, y_train, y_test)
    bayes.do_bayes()

def skleran_neural(X_train, X_test, y_train, y_test):

    print("-----------------SKLEARN NEURAL------------------------")
    bayes = SKlearnNeural(X_train, X_test, y_train, y_test)
    bayes.do_SVM()


def reccurent_neural(X_train, X_test, y_train, y_test):
    # print("-----------------RECURRENT NEURAL------------------------")
    recurent_neural = ReccurentNeuralMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                            np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    recurent_neural.do_rc()

def voiting_classifier(X_train, X_test, y_train, y_test):
    # print("-----------------VOITING CLASSIFIER------------------------")
    voiting = VoitingClassifierMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                            np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    voiting.do_voiting()



#***************************************************************************
test_size = 0.25
blocks_size = 200
max_features = 500
start_time = datetime.datetime.now()

prepare_text(max_features=max_features)
# prepare_text2(blocks_size)
# X_train, X_test, y_train, y_test = TextPreparation.get_dataset_new()
X_train, X_test, y_train, y_test = divide_dataset(test_size=test_size)

logistic_regression(X_train, X_test, y_train, y_test)
# random_forests(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)
naive_bayes(scale(X_train), scale(X_test), y_train, y_test)
# skleran_neural(X_train, X_test, y_train, y_test)
# reccurent_neural(X_train, X_test, y_train, y_test)









voiting_classifier(X_train, X_test, y_train, y_test)



print("*****************************************************")
print("TIME: {}".format(datetime.datetime.now() - start_time))

from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod
from methods.naive_bayes import NaiveBayesMethod
from methods.reccurent_neural import ReccurentNeuralMethod
from methods.sklearn_neural import SKlearnNeural
from methods.voting_classifier import VoitingClassifierMethod
from new_preprocessing import *


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

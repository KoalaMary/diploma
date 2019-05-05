from preprocessing import TextPreparation
from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod
from methods.naive_bayes import NaiveBayesMethod
from methods.reccurent_neural import ReccurentNeuralMethod
from sklearn.model_selection import train_test_split
import datetime
import numpy as np


def divide_dataset(test_size):
    x, y = TextPreparation.get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


test_size = 0.15
blocks_size = 200
start_time = datetime.datetime.now()

# ex = TextPreparation()
# ex.create_all_dicts(blocks_size=blocks_size)
X_train, X_test, y_train, y_test = divide_dataset(test_size=test_size)
# print(len(X_train[0]), len(X_train[1]))


# lr_solvers = ["newton-cg", "sag", "saga", "lbfgs"]
solver = "newton-cg"
# for solver in lr_solvers:
print("-----------------LOGISTIC REGRESSION: {}------------------------".format(solver))
lr = LogisticRegressionMethod(X_train, X_test, y_train, y_test, solver=solver)
lr.do_LR()

print("-----------------RANDOM FORESTS------------------------")
rf = RandomForestMethod(X_train, X_test, y_train, y_test)
rf.do_RF()

print("-----------------SVM METHOD------------------------")
svm = SVMMethod(X_train, X_test, y_train, y_test)
svm.do_SVM()

print("-----------------KNN------------------------")
knn = KNNMethod(X_train, X_test, y_train, y_test)
knn.do_knn()

print("-----------------NAIVE BAYES------------------------")
bayes = NaiveBayesMethod(X_train, X_test, y_train, y_test)
bayes.do_bayes()

# print("-----------------RECURRENT NEURAL------------------------")
# recurent_neural = ReccurentNeuralMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
#                                         np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
# # recurent_neural = ReccurentNeuralMethod()
# recurent_neural.do_rc()

print("*****************************************************")
print("TIME: {}".format(datetime.datetime.now() - start_time))

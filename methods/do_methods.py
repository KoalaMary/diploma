from methods.my_logistic_regression import LogisticRegressionMethod
from methods.random_forest import RandomForestMethod
from methods.svm import SVMMethod
from methods.knn import KNNMethod
from methods.naive_bayes import NaiveBayesMethod
from methods.reccurent_neural import ReccurentNeuralMethod
from methods.sklearn_neural import SKlearnNeural
from methods.voting_classifier import VoitingClassifierMethod
from new_preprocessing import *
from reports.plots import plot_confusion_matrix


def logistic_regression(X_train, X_test, y_train, y_test, labelencoder):
    # lr_solvers = ["newton-cg", "sag", "saga"]
    solver = "newton-cg"
    # for solver in lr_solvers:
    print("-----------------LOGISTIC REGRESSION: {}------------------------".format(solver))
    # lr = LogisticRegressionMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
    #                               np.array(y_train).astype("float32"), np.array(y_test).astype("float32"),
    #                               solver=solver)
    lr = LogisticRegressionMethod(X_train, X_test, y_train, y_test)
    y_pred_lr = lr.do_LR()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_lr, normalize=True,
                                      title=f'Logistic regression: {solver}', labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('log_reg.png')


def random_forests(X_train, X_test, y_train, y_test, labelencoder):
    print("-----------------RANDOM FORESTS------------------------")
    rf = RandomForestMethod(X_train, X_test, y_train, y_test)
    y_pred_rf = rf.do_RF()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_rf, normalize=True, title=f'Random forests',
                                      labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('random_for.png')


def svm(X_train, X_test, y_train, y_test, labelencoder):
    print("-----------------SVM METHOD------------------------")
    svm = SVMMethod(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
    y_pred_svm = svm.do_SVM()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_svm, normalize=True, title=f'SVM',
                                      labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('svm.png')


def knn(X_train, X_test, y_train, y_test, labelencoder):
    print("-----------------KNN------------------------")
    knn = KNNMethod(X_train, X_test, y_train, y_test)
    y_pred_knn = knn.do_knn()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_knn, normalize=True, title=f'KNN',
                                      labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('knn.png')


def naive_bayes(X_train, X_test, y_train, y_test, labelencoder):
    print("-----------------NAIVE BAYES------------------------")
    bayes = NaiveBayesMethod(X_train, X_test, y_train, y_test)
    y_pred_bayes = bayes.do_bayes()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_bayes, normalize=True, title=f'Naive bayes',
                                      labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('bayes.png')


def skleran_neural(X_train, X_test, y_train, y_test, labelencoder):
    print("-----------------SKLEARN NEURAL------------------------")
    bayes = SKlearnNeural(X_train, X_test, y_train, y_test)
    y_pred_neural = bayes.do_SVM()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_neural, normalize=True,
                                      title=f'Multi-layer Perceptron', labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('neural.png')


def reccurent_neural(X_train, X_test, y_train, y_test, labelencoder):
    # print("-----------------RECURRENT NEURAL------------------------")
    recurent_neural = ReccurentNeuralMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                            np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    recurent_neural.do_rc()


def voiting_classifier(X_train, X_test, y_train, y_test, labelencoder):
    # print("-----------------VOITING CLASSIFIER------------------------")
    voiting = VoitingClassifierMethod(np.array(X_train).astype("float32"), np.array(X_test).astype("float32"),
                                      np.array(y_train).astype("float32"), np.array(y_test).astype("float32"))
    # recurent_neural = ReccurentNeuralMethod()
    y_pred_voitiig = voiting.do_voiting()
    fig_bayes = plot_confusion_matrix(y_true=y_test, y_pred=y_pred_voitiig, normalize=True, title=f'Voting classifier',
                                      labelencoder=labelencoder)
    fig_bayes.show()
    fig_bayes.savefig('voiting .png')

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation
from sklearn.multiclass import OneVsRestClassifier

class SKlearnNeural:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_SVM(self):
        # Training
        clf = MLPClassifier()
        clf.fit(self.X_train, self.y_train)

        # Prediction
        y_pred = clf.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, y_pred)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        print("svm_accuracy: {}".format(svm_accuracy))

        return y_pred
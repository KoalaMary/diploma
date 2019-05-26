from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score


# NEED TO BE COMPLITED!!!

class SVMMethod:
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
        # x, y = TextPreparation.get_dataset()
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.text_size, random_state=42)
        # svm = SVC(kernel='linear', C=1, probability=True).fit(self.X_train, self.y_train)
        clf = OneVsRestClassifier(LinearSVC(multi_class="ovr"))
        clf.fit(self.X_train, self.y_train)

        # Prediction
        y_pred = clf.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, y_pred)

        # Cross validation
        # cross_validation = cross_val_score(clf, self.X_train, self.y_train, cv=4)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        print("svm_accuracy: {}".format(svm_accuracy))
        # print("Cross validation: {}".format(cross_validation))
        # print("RES: {}".format(res))

        return y_pred

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation


# NEED TO BE COMPLITED!!!

class SVMMethod:

    def do_SVM(self):
        # Training
        x, y = TextPreparation.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)

        # Prediction
        y_pred = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, y_pred)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(y_test))
        print("svm_accuracy: {}".format(svm_accuracy))
        # print("RES: {}".format(res))

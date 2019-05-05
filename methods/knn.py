from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation


class KNNMethod:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_knn(self):
        # Training
        # x, y = TextPreparation.get_dataset()
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto").fit(self.X_train, self.y_train)

        # Prediction
        y_pred = knn.predict(self.X_test)
        knn_accuracy = accuracy_score(self.y_test, y_pred)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        print("knn_accuracy: {}".format(knn_accuracy))

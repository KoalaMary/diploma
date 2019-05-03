from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation


class KNNMethod:

    def do_knn(self):
        # Training
        x, y = TextPreparation.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        knn = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)

        # Prediction
        y_pred = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, y_pred)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(y_test))
        print("knn_accuracy: {}".format(knn_accuracy))

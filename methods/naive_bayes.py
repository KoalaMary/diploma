from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import TextPreparation


class NaiveBayesMethod:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_bayes(self):
        # Training
        # x, y = TextPreparation.get_dataset()
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.text_size)
        clf = GaussianNB()
        model = clf.fit(self.X_train, self.y_train)

        # Prediction
        y_pred = model.predict(self.X_test)
        res = accuracy_score(self.y_test, y_pred)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        # print("Y TRAIN: {}".format(y_train))
        print("RES: {}".format(res))

        return y_pred

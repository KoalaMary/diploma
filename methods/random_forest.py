from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from preprocessing import TextPreparation

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class RandomForestMethod:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_RF(self):
        # Training
        # x, y = TextPreparation.get_dataset()
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.text_size)
        n_est = 100
        max_d = 5
        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_est, max_depth=max_d)
        rfc.fit(self.X_train, self.y_train)

        # Prediction
        y_pred = rfc.predict(self.X_test)
        accuracy_forest = rfc.score(self.X_test, self.y_test)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        print("accuracy_forest: {}".format(accuracy_forest))

        return y_pred
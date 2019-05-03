import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import TextPreparation
import os


class LogisticRegressionMethod:

    def do_LR(self):
        # Training
        x, y = TextPreparation.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
        model = clf.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        res = model.score(X_test, y_test)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(y_test))
        print("Y TRAIN: {}".format(y_train))
        print("RES: {}".format(res))

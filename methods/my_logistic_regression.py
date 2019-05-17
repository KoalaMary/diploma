from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from preprocessing import TextPreparation


class LogisticRegressionMethod:
    solver = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test, solver="newton-cg"):
        self.solver = solver
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def do_LR(self):
        # Training
        # x, y = TextPreparation.get_dataset()
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.text_size)
        clf = LogisticRegression(random_state=42, solver=self.solver, multi_class="multinomial")
        model = clf.fit(self.X_train, self.y_train)

        # Prediction
        y_pred = model.predict(self.X_test)
        res = model.score(self.X_test, self.y_test)

        print("Y PREd: {}".format(y_pred))
        print("Y TEST: {}".format(self.y_test))
        print("Y TRAIN: {}".format(self.y_train))
        print("RES: {}".format(res))

        return y_pred

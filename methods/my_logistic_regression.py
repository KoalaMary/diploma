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

        # ROC
        # logistic_accuracy = accuracy_score(y_test, y_pred)
        # metrics_for_forest = precision_recall_fscore_support(y_test, y_pred, average="macro")
        # probs = clf.predict_proba(X_test)
        # preds = probs[:, 1]
        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        # roc_auc = metrics.auc(fpr, tpr)
        # plt.title("Receiver Operating Characteristic")
        # plt.plot(fpr, tpr, "green", label="AUC = %0.2f" % roc_auc)
        # plt.legend(loc="lower right")
        # plt.plot([0, 1], [0, 1], "r--")
        # plt.xlim([-0.1, 1])
        # plt.ylim([-0.1, 1.1])
        # plt.ylabel("True Positive Rate")
        # plt.xlabel("False Positive Rate" )
        #
        # file_name = "lr_res_{}.png".format(self.solver)
        # plt.savefig(file_name)

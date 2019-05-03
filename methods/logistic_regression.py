import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os


class LR(object):
    # def __init__(self, split):
    #     self.split = split

    def get_dataset(self):
        train_path = os.path.join(os.getcwd(), "..", "train.csv")
        data = pd.read_csv(train_path)
        dataset = data.values.tolist()
        shuffle(dataset)
        x = []
        y = []
        for row in dataset:
            x.append(row[:-1])
            y.append(row[-1])

        return x, y

    def lr(self):
        x, y = self.get_dataset()

        # Training
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model = LogisticRegression().fit(X_train, y_train)

        # Prediction
        x_new, y_new = self.get_dataset()
        y_pred = model.predict(x_new)
        logistic_accuracy = accuracy_score(y_new, y_pred)
        metrics_for_forest = precision_recall_fscore_support(y_new, y_pred, average='macro')
        print('\nacc log_reg: ' + str(logistic_accuracy))
        print('recall: ' + str(metrics_for_forest[1]))
        print('precision: ' + str(metrics_for_forest[0]))
        print('f_score: ' + str(metrics_for_forest[2]))

        print('\nacc log_reg: ' + str(logistic_accuracy))
        print('recall: ' + str(metrics_for_forest[1]))
        print('precision: ' + str(metrics_for_forest[0]))
        print('f_score: ' + str(metrics_for_forest[2]))

        probs = model.predict_proba(x_new)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_new, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'green', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid(True)
        plt.savefig('result.png')

        return logistic_accuracy

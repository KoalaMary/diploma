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
        # print("RES: {}".format(res))

        return y_pred


        # cm = confusion_matrix(self.y_test, y_pred)
        # # Only use the labels that appear in the data
        # classes = unique_labels(self.y_test, y_pred)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
        #
        #
        # print(cm)
        #
        # fig, ax = plt.subplots()
        # im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # ax.figure.colorbar(im, ax=ax)
        # # We want to show all ticks...
        # ax.set(xticks=np.arange(cm.shape[1]),
        #        yticks=np.arange(cm.shape[0]),
        #        # ... and label them with the respective list entries
        #        xticklabels=classes, yticklabels=classes,
        #        title="example",
        #        ylabel='True label',
        #        xlabel='Predicted label')
        #
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")
        #
        # # Loop over data dimensions and create text annotations.
        # fmt = '.2f'
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         ax.text(j, i, format(cm[i, j], fmt),
        #                 ha="center", va="center",
        #                 color="white" if cm[i, j] > thresh else "black")
        # fig.tight_layout()
        # plt.show()
        # return ax
